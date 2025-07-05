"""
Main ingestion script for processing patent JSON documents.

- Chunks text fields for the Vector DB.
- Uses the full JSON for the Knowledge Graph.
"""

import os
import asyncio
import logging
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

import asyncpg
from dotenv import load_dotenv

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder
from .graph_builder import create_graph_builder

# Import agent utilities
try:
    from ..agent.db_utils import initialize_database, close_database, db_pool
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult
except ImportError:
    # For direct execution or testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.db_utils import initialize_database, close_database, db_pool
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting patent documents into vector DB (via chunking) and knowledge graph (full object)."""
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing patent JSON documents
            clean_before_ingest: Whether to clean existing data before ingestion
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        
        # Initialize components - Chunker is restored for VectorDB part
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )
        
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing ingestion pipeline...")
        await initialize_database()
        await initialize_graph()
        await self.graph_builder.initialize()
        self._initialized = True
        logger.info("Ingestion pipeline initialized")
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await self.graph_builder.close()
            await close_graph()
            await close_database()
            self._initialized = False
    
    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest all patent documents from the documents folder.
        """
        if not self._initialized:
            await self.initialize()
        
        if self.clean_before_ingest:
            await self._clean_databases()
        
        # Find all JSON files
        json_files = self._find_json_files()
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.documents_folder}")
            return []
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        results = []
        for i, file_path in enumerate(json_files):
            try:
                logger.info(f"Processing file {i+1}/{len(json_files)}: {file_path}")
                result = await self._ingest_single_document(file_path)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(json_files))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                results.append(IngestionResult(
                    document_id="", title=os.path.basename(file_path),
                    chunks_created=0, entities_extracted=0, relationships_created=0,
                    processing_time_ms=0, errors=[str(e)]
                ))
        
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)
        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} text chunks, {total_errors} errors")
        
        return results
    
    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """Ingest a single patent document, chunking its text for the vector DB."""
        start_time = datetime.now()
        
        patent_data = self._read_document(file_path)
        document_title = self._extract_title(patent_data, file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        
        document_metadata = self._extract_document_metadata(patent_data, file_path)
        # We add the full patent data to the metadata so the graph builder can access it
        document_metadata['full_patent_data'] = patent_data
        
        logger.info(f"Processing patent: {document_title}")
        
        # --- VectorDB Processing ---
        # 1. Concatenate text fields from the patent for chunking.
        text_content = "\n\n".join(filter(None, [
            f"Title: {patent_data.get('title', '')}",
            f"Abstract: {patent_data.get('abstract', '')}",
            f"Background: {patent_data.get('background', '')}",
            f"Summary: {patent_data.get('summary', '')}",
            f"Claims: {patent_data.get('claims', '')}",
            f"Full Description: {patent_data.get('full_description', '')}"
        ]))

        # 2. Chunk the concatenated text as per original logic.
        chunks = await self.chunker.chunk_document(
            content=text_content, title=document_title, source=document_source, metadata=document_metadata
        )
        if not chunks:
            logger.warning(f"No text chunks created for {document_title}")

        logger.info(f"Created {len(chunks)} text chunks for vector DB")
        
        # 3. Generate embeddings for each text chunk.
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # 4. Save document and all its text chunks with embeddings to PostgreSQL.
        document_id = await self._save_to_postgres(
            document_title, document_source, patent_data, embedded_chunks, document_metadata
        )
        logger.info(f"Saved document and {len(chunks)} text chunks to PostgreSQL with Doc ID: {document_id}")
        
        # --- Knowledge Graph Processing ---
        relationships_created = 0
        graph_errors = []
        if not self.config.skip_graph_building:
            try:
                logger.info("Building knowledge graph relationships from full JSON object...")
                # The graph builder will use the 'full_patent_data' from the metadata
                graph_result = await self.graph_builder.add_document_to_graph(
                    chunks=embedded_chunks, # Pass chunks for context if needed
                    document_title=document_title,
                    document_source=document_source,
                    document_metadata=document_metadata
                )
                relationships_created = graph_result.get("episodes_created", 0)
                graph_errors = graph_result.get("errors", [])
                logger.info(f"Added {relationships_created} episodes to knowledge graph")
            except Exception as e:
                error_msg = f"Failed to add to knowledge graph: {str(e)}"
                logger.error(error_msg, exc_info=True)
                graph_errors.append(error_msg)
        else:
            logger.info("Skipping knowledge graph building.")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResult(
            document_id=document_id, title=document_title, chunks_created=len(chunks),
            entities_extracted=0, relationships_created=relationships_created,
            processing_time_ms=processing_time, errors=graph_errors
        )

    def _find_json_files(self) -> List[str]:
        """Find all JSON files in the documents folder."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []
        return sorted(glob.glob(os.path.join(self.documents_folder, "**", "*.json"), recursive=True))

    def _read_document(self, file_path: str) -> Dict[str, Any]:
        """Read patent data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_title(self, content: Dict[str, Any], file_path: str) -> str:
        """Extract title from patent data or filename."""
        return content.get("title", os.path.splitext(os.path.basename(file_path))[0])

    def _extract_document_metadata(self, content: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Extract metadata from patent data."""
        return {
            "file_path": file_path, "ingestion_date": datetime.now().isoformat(),
            "application_number": content.get("application_number"),
            "publication_number": content.get("publication_number"),
        }

    async def _save_to_postgres(
        self, title: str, source: str, content: Dict[str, Any], chunks: List[DocumentChunk], metadata: Dict[str, Any]
    ) -> str:
        """Save document and its text chunks to PostgreSQL."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                document_result = await conn.fetchrow(
                    "INSERT INTO documents (title, source, content, metadata) VALUES ($1, $2, $3, $4) RETURNING id::text",
                    title, source, json.dumps(content), json.dumps(metadata)
                )
                document_id = document_result["id"]
                
                # Insert all text chunks for the document
                for chunk in chunks:
                    embedding_data = '[' + ','.join(map(str, chunk.embedding)) + ']' if hasattr(chunk, 'embedding') and chunk.embedding else None
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id, chunk.content, embedding_data,
                        chunk.index, json.dumps(chunk.metadata), chunk.token_count
                    )
                return document_id

    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("TRUNCATE TABLE messages, sessions, chunks, documents RESTART IDENTITY")
        logger.info("Cleaned PostgreSQL database")
        
        await self.graph_builder.clear_graph()
        logger.info("Cleaned knowledge graph")


async def main():
    """Main function for running patent ingestion."""
    parser = argparse.ArgumentParser(description="Ingest patent JSON documents into vector DB and knowledge graph")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean existing data before ingestion")
    # Restoring original chunking arguments
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting document text")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size for text")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking for text")
    parser.add_argument("--fast", "-f", action="store_true", help="Fast mode: skip knowledge graph building")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Restoring original IngestionConfig
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
        extract_entities=False, # Handled by Graphiti
        skip_graph_building=args.fast
    )
    
    pipeline = DocumentIngestionPipeline(config=config, documents_folder=args.documents, clean_before_ingest=args.clean)
    
    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")
    
    try:
        start_time = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        total_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*50 + "\nINGESTION SUMMARY\n" + "="*50)
        print(f"Documents processed: {len(results)}")
        print(f"Total text chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total graph episodes: {sum(r.relationships_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds\n")
        
        for result in results:
            status = "✓" if not result.errors else "✗"
            print(f"{status} {result.title}: {result.chunks_created} chunks, {result.relationships_created} episodes")
            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
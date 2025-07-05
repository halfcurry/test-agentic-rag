"""
Knowledge graph builder for extracting patent entities and relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from pydantic import BaseModel, Field
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    # For direct execution or testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# --- Custom Patent Entity and Edge Types ---

class Patent(BaseModel):
    """A patent document, application, or publication."""
    publication_number: Optional[str] = Field(None, description="The publication number of the patent (e.g., US20180232870A1).")
    application_number: Optional[str] = Field(None, description="The application number for the patent filing.")
    filing_date: Optional[datetime] = Field(None, description="The date the patent was filed.")
    publication_date: Optional[datetime] = Field(None, description="The date the patent was published.")
    status: Optional[str] = Field(None, description="The current legal status of the patent (e.g., PENDING, GRANTED, ABANDONED).")

class Inventor(BaseModel):
    """An inventor listed on a patent."""
    full_name: Optional[str] = Field(None, description="The full name of the inventor.")
    city: Optional[str] = Field(None, description="The city where the inventor resides.")
    country: Optional[str] = Field(None, description="The country where the inventor resides (e.g., KR, US).")

class Examiner(BaseModel):
    """A patent examiner who reviewed the patent application."""
    full_name: Optional[str] = Field(None, description="The full name of the patent examiner.")
    examiner_id: Optional[str] = Field(None, description="The unique identifier for the examiner.")

class Classification(BaseModel):
    """A classification code (CPC, IPCR, USPC) assigned to a patent."""
    code: Optional[str] = Field(None, description="The classification code itself (e.g., G06T7/001).")
    type: Optional[str] = Field(None, description="The type of classification system (e.g., CPC, IPCR, USPC).")

class Technology(BaseModel):
    """A technology, method, or concept described in the patent."""
    domain: Optional[str] = Field(None, description="The specific technical field or domain of the technology.")


# --- Custom Edge Types ---

class INVENTED_BY(BaseModel):
    """The relationship between a Patent and an Inventor."""
    role: str = Field("Inventor", description="The role of the person in the invention.")

class EXAMINED_BY(BaseModel):
    """The relationship between a Patent and an Examiner."""
    role: str = Field("Examiner", description="The role of the person in reviewing the patent.")

class CLASSIFIED_AS(BaseModel):
    """The relationship between a Patent and a Classification."""
    is_main: bool = Field(False, description="Indicates if this is the main classification for the patent.")

class DESCRIBES_TECHNOLOGY(BaseModel):
    """The relationship between a Patent and a Technology it describes."""
    novelty: Optional[str] = Field(None, description="A brief on the novelty of the technology as per the patent abstract or claims.")


class GraphBuilder:
    """Builds knowledge graph from patent documents."""

    def __init__(self):
        """Initialize graph builder with patent ontology."""
        self.graph_client = GraphitiClient()
        self._initialized = False

        # Define the custom ontology for Graphiti
        self.entity_types = {
            "Patent": Patent,
            "Inventor": Inventor,
            "Examiner": Examiner,
            "Classification": Classification,
            "Technology": Technology,
        }
        self.edge_types = {
            "INVENTED_BY": INVENTED_BY,
            "EXAMINED_BY": EXAMINED_BY,
            "CLASSIFIED_AS": CLASSIFIED_AS,
            "DESCRIBES_TECHNOLOGY": DESCRIBES_TECHNOLOGY,
        }
        self.edge_type_map = {
            ("Patent", "Inventor"): ["INVENTED_BY"],
            ("Patent", "Examiner"): ["EXAMINED_BY"],
            ("Patent", "Classification"): ["CLASSIFIED_AS"],
            ("Patent", "Technology"): ["DESCRIBES_TECHNOLOGY"],
        }

    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True

    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False

    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a full patent document to the knowledge graph as a single episode.
        
        Args:
            chunks: List of text chunks (for context only).
            document_title: Title of the patent.
            document_source: Source file of the patent.
            document_metadata: Additional metadata which MUST contain the full patent JSON.
        
        Returns:
            Processing results.
        """
        if not self._initialized:
            await self.initialize()
        
        # ⬇️ CORRECTED LOGIC: Retrieve the full patent data from the metadata dictionary.
        if document_metadata is None:
            return {"episodes_created": 0, "errors": ["Metadata with patent data not provided."]}
        
        patent_data = document_metadata.get("full_patent_data")
        
        if not isinstance(patent_data, dict):
            return {"episodes_created": 0, "errors": ["'full_patent_data' not found or is not a valid JSON object in metadata."]}

        logger.info(f"Adding patent to knowledge graph: {document_title}")
        episodes_created = 0
        errors = []

        try:
            episode_id = patent_data.get("publication_number", document_source)

            # ⬇️ NOW CORRECTLY USES `patent_data` AS THE EPISODE BODY
            await self.graph_client.add_episode(
                name=episode_id,
                episode_body=json.dumps(patent_data),
                source=EpisodeType.text,
                source_description=f"Patent Application: {document_title}",
                reference_time=datetime.fromisoformat(patent_data.get("date_published", datetime.now().isoformat())),
                # entity_types=self.entity_types,
                # edge_types=self.edge_types,
                # edge_type_map=self.edge_type_map
            )
            episodes_created = 1
            logger.info(f"✓ Added episode {episode_id} to knowledge graph")

        except Exception as e:
            error_msg = f"Failed to add patent {document_title} to graph: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
        logger.info(f"Graph building complete for {document_title}: {episodes_created} episodes created, {len(errors)} errors")
        return result

    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        This function is a pass-through. Entity extraction is handled by Graphiti.
        """
        logger.info("Skipping manual entity extraction; handled by Graphiti for JSON episodes.")
        return chunks

    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()

        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")

def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()
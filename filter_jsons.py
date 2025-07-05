import os
import argparse
import shutil
import concurrent.futures
import orjson  # Using the faster orjson library

def process_file(filename, input_path, output_path):
    """
    Processes a single JSON file using the high-performance orjson library.
    This function is designed to be run in a separate process.

    Returns:
        A tuple of (status, filename). Status can be 'COPIED', 'SKIPPED', or 'ERROR'.
    """
    json_filepath = os.path.join(input_path, filename)
    try:
        # Open the file in binary mode ('rb') and use orjson for faster parsing
        with open(json_filepath, 'rb') as f:
            json_data = orjson.loads(f.read())

        if json_data.get('decision') in ["ACCEPTED", "REJECTED"]:
            destination_filepath = os.path.join(output_path, filename)
            shutil.copy2(json_filepath, destination_filepath)
            return ('COPIED', filename)
        else:
            return ('SKIPPED', filename)

    except Exception as e:
        # Any error during file processing is caught here
        print(f"Error processing file '{filename}': {e}")
        return ('ERROR', filename)

def main():
    """
    Main function to parse arguments and coordinate the parallel processing of JSON files.
    """
    parser = argparse.ArgumentParser(
        description="Rapidly copy JSON files based on a decision field using parallel processing and orjson."
    )
    parser.add_argument(
        "max_files",
        type=int,
        help="Maximum number of JSON files to copy to the output directory."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing JSON files."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where accepted or rejected JSON files will be copied."
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_folder)
    output_path = os.path.abspath(args.output_folder)
    max_files_to_copy = args.max_files

    os.makedirs(output_path, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"🚀 Starting high-speed processing with orjson...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Max files to copy: {max_files_to_copy}")

    try:
        with os.scandir(input_path) as it:
            files = sorted([entry.name for entry in it if entry.is_file() and entry.name.endswith('.json')])

        if not files:
            print("No JSON files found in the input directory.")
            return

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, f, input_path, output_path): f for f in files}

            for future in concurrent.futures.as_completed(future_to_file):
                if copied_count >= max_files_to_copy:
                    for f in future_to_file:
                        if not f.done() and not f.running():
                            f.cancel()
                    break

                status, filename = future.result()
                if status == 'COPIED':
                    copied_count += 1
                    print(f"✅ Copied '{filename}' ({copied_count}/{max_files_to_copy})")
                elif status == 'SKIPPED':
                    skipped_count += 1
                elif status == 'ERROR':
                    error_count += 1
            
            if copied_count >= max_files_to_copy:
                 print(f"\nReached maximum file copy limit of {max_files_to_copy}. Halting.")


    except FileNotFoundError:
        print(f"Error: Input folder '{input_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    total_processed = copied_count + skipped_count + error_count
    print(f"\n🏁 Finished. Processed {total_processed} files. Copied {copied_count}, Skipped {skipped_count}, Errors {error_count}.")


if __name__ == "__main__":
    main()
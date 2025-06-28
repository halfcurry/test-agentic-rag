import os
import json
import argparse

def convert_json_to_markdown(json_data, indent=0):
    """
    Converts a JSON object or array into a Markdown formatted string.
    """
    markdown_output = []
    indent_str = "    " * indent

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            # For top-level keys, use a heading. For nested keys, use bold.
            if indent == 0:
                markdown_output.append(f"{indent_str}# {key.replace('_', ' ').title()}\n")
            else:
                markdown_output.append(f"{indent_str}**{key.replace('_', ' ').title()}:** ")

            if isinstance(value, (dict, list)):
                # If the value is a nested object or list, add a newline and recurse
                markdown_output.append("\n")
                markdown_output.append(convert_json_to_markdown(value, indent + 1))
            else:
                # Otherwise, append the value directly
                markdown_output.append(f"{value}\n")
    elif isinstance(json_data, list):
        for item in json_data:
            markdown_output.append(f"{indent_str}- ") # List item prefix
            if isinstance(item, (dict, list)):
                # If the list item is a nested object or list, add a newline and recurse
                markdown_output.append("\n")
                markdown_output.append(convert_json_to_markdown(item, indent + 1))
            else:
                markdown_output.append(f"{item}\n")
    else:
        # For simple values (strings, numbers, booleans)
        markdown_output.append(f"{json_data}\n")

    return "".join(markdown_output)

def main():
    """
    Main function to parse arguments, read JSON files, and convert them to Markdown.
    """
    parser = argparse.ArgumentParser(
        description="Convert JSON files to Markdown files."
    )
    parser.add_argument(
        "max_files",
        type=int,
        help="Maximum number of JSON files to process."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing JSON files."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where Markdown files will be saved."
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_folder)
    output_path = os.path.abspath(args.output_folder)
    max_files_to_process = args.max_files

    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    processed_count = 0
    print(f"Processing JSON files from: {input_path}")
    print(f"Saving Markdown files to: {output_path}")
    print(f"Maximum files to process: {max_files_to_process}")

    try:
        # List all files in the input folder
        files = [f for f in os.listdir(input_path) if f.endswith('.json')]
        files.sort() # Ensure consistent order

        for filename in files:
            if processed_count >= max_files_to_process:
                print(f"Reached maximum file limit of {max_files_to_process}. Stopping.")
                break

            json_filepath = os.path.join(input_path, filename)
            markdown_filename = os.path.splitext(filename)[0] + ".md"
            markdown_filepath = os.path.join(output_path, markdown_filename)

            print(f"Converting '{filename}' to '{markdown_filename}'...")

            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                markdown_content = convert_json_to_markdown(json_data)

                with open(markdown_filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                processed_count += 1
                print(f"Successfully converted '{filename}'.")

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{filename}'. Skipping.")
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while processing '{filename}': {e}. Skipping.")

    except FileNotFoundError:
        print(f"Error: Input folder '{input_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\nFinished processing. Converted {processed_count} JSON files to Markdown.")

if __name__ == "__main__":
    main()

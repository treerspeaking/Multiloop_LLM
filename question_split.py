import json
import os

def split_json_file(input_file_path, output_dir, questions_per_file=75):
    """
    Parses a large JSON file containing a list of questions into smaller JSON files.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_dir (str): The directory where the smaller JSON files will be saved.
        questions_per_file (int): The number of questions to include in each smaller file.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'. Ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    if not isinstance(all_questions, list):
        print("Error: The JSON file should contain a list of questions at the root level.")
        return

    if not all_questions:
        print("The input JSON file is empty. No output files will be created.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_questions = len(all_questions)
    num_output_files = 0

    for i in range(0, num_questions, questions_per_file):
        chunk = all_questions[i:i + questions_per_file]
        output_file_name = f"output_part_{num_output_files + 1}.json"
        output_file_path = os.path.join(output_dir, output_file_name)

        try:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(chunk, outfile, indent=4)
            print(f"Successfully created '{output_file_path}' with {len(chunk)} questions.")
            num_output_files += 1
        except Exception as e:
            print(f"An error occurred while writing to '{output_file_path}': {e}")

    if num_output_files > 0:
        print(f"\nSuccessfully split the input file into {num_output_files} smaller JSON files in the '{output_dir}' directory.")
    else:
        print("No output files were created.")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy input JSON file for demonstration
    # dummy_data = []
    # for i in range(200): # Example: 200 questions
    #     dummy_data.append({
    #         "query": f"This is question number {i+1}?",
    #         "answer": f"This is the answer to question {i+1}.",
    #         "question_type": "example_query",
    #         "evidence_list": []
    #     })

    input_json_file = "MultiHopRAG (1).json"

    # Define the output directory
    output_directory = "output_chunks"

    # Call the function to split the JSON
    split_json_file(input_json_file, output_directory, questions_per_file=75)

    # To use this with your actual file:
    # 1. Save the code above as a Python file (e.g., split_json.py).
    # 2. Make sure your large JSON file (e.g., "your_large_file.json") is in the same directory
    #    or provide the correct path to it.
    # 3. Run the script from your terminal: python split_json.py
    # 4. Then, uncomment and modify the following lines in the `if __name__ == "__main__":` block:
    #
    # actual_input_file = "your_large_file.json" # Replace with your actual file name
    # actual_output_dir = "split_output_files"
    # print(f"\nProcessing your actual file: '{actual_input_file}'")
    # split_json_file(actual_input_file, actual_output_dir, questions_per_file=75)
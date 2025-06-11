from utils.utils import split_json_file


input_json_file = "MultiHopRAG (1).json"

# Define the output directory
output_directory = "output_chunks"

# Call the function to split the JSON
split_json_file(input_json_file, output_directory, questions_per_file=100)

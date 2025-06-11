from utils.utils import count_question_types


# Replace 'data.json' with your actual file path
file_path = '/home/treerspeaking/src/python/Multiloop_LLM/MultiHopRAG (1).json'
question_type_counts = count_question_types(file_path)

# You can use the counts for further processing if needed
print("\nTotal questions analyzed:", sum(question_type_counts.values()))
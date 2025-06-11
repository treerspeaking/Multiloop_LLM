from utils.val import compare_result_with_f1

output_file = "/home/treerspeaking/src/python/Multiloop_LLM/react_rag_output.json"
ques_file = "/home/treerspeaking/src/python/Multiloop_LLM/input_test.json"

compare_result_with_f1(output_file, ques_file)
# ReAct

This is the simple implementation of the ReAct Pipeline.

To run the repository Follow the following step

## 1

Export the Google Gemini key which are available for free through the [Google AI studio](https://aistudio.google.com/apikey)

```bash
export GOOGLE_API_KEY="YOUR API KEY"
```

## 2 

To generate the output prediction, run the pipeline_ReAct.py

```bash
python pipeline_ReAct.py
```

The result will be output in react_rag_output.json

## 3 

Finally to evaluate the result run

```bash
python compare_result.py
```

The detail of the result will be output in detailed_evaluation_results.json, In addition, there will also be wrong_answers_with_f1.json where all the wrong result are output for further inspection in order to propose the new methods.
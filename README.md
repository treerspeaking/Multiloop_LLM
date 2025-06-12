# ReAct

This is the simple implementation of the ReAct Pipeline.

To run the repository Follow the following step:

## 1

Export the Google Gemini key which are available for free through the [Google AI studio](https://aistudio.google.com/apikey)

```bash
export GOOGLE_API_KEY="YOUR API KEY"
```

## 2

Create the required environment.

```bash
conda env create -f environment.yml
```

Activate the environment.

```bash
conda activate multihop
```

## 3

To generate the output prediction, run the pipeline_ReAct.py

```bash
python pipeline_ReAct.py
```

The result will be output in react_rag_output.json

## 4

Finally to evaluate the result run

```bash
python compare_result.py
```

The detail of the result will be output in detailed_evaluation_results.json, In addition, there will also be wrong_answers_with_f1.json where all the wrong result are output for further inspection in order to propose the new methods.

The final result is:
exact_match_accuracy": 0.58 <br>
exact_matches": 58 <br>
average_precision": 0.595 <br>
average_recall": 0.5933333333333333 <br>
average_f1_score": 0.5916666666666667<br>
average_iter_count": 2.41 <br>
incorrect_count": 42
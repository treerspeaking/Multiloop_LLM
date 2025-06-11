import json
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

def normalize_answer(answer: str) -> str:
    """
    Normalize answer text for comparison by:
    - Converting to lowercase
    - Removing extra whitespace
    - Removing punctuation
    - Handling special cases
    """
    if not answer or answer.strip().lower() == "insufficient information":
        return "insufficient information"
    
    # Convert to lowercase and strip
    answer = answer.lower().strip()
    
    # Remove common punctuation but keep spaces
    answer = re.sub(r'[^\w\s]', ' ', answer)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer

def get_tokens(text: str) -> List[str]:
    """
    Tokenize text into words for F1 calculation.
    """
    normalized = normalize_answer(text)
    return normalized.split()

def calculate_f1_score(predicted: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate F1 score between predicted and ground truth answers.
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    pred_tokens = get_tokens(predicted)
    gt_tokens = get_tokens(ground_truth)
    
    # Handle edge cases
    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0
    
    # Count token frequencies
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # Calculate overlap
    overlap = sum((pred_counter & gt_counter).values())
    
    # Calculate precision and recall
    precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = overlap / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_exact_match(predicted: str, ground_truth: str) -> bool:
    """
    Calculate exact match after normalization.
    """
    return normalize_answer(predicted) == normalize_answer(ground_truth)

def compare_result_with_f1(output_file: str, multi_hop_file: str, detailed_output: bool = True):
    """
    Enhanced comparison function that calculates both exact match accuracy and F1 scores.
    
    Args:
        output_file: Path to the output JSON file with LLM answers
        multi_hop_file: Path to the ground truth JSON file
        detailed_output: Whether to save detailed results for incorrect answers
    """
    
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        outputs = json.load(f)
        
    with open(multi_hop_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    if len(outputs) != len(ground_truth):
        print(f"Warning: Mismatch in number of items. Outputs: {len(outputs)}, Ground Truth: {len(ground_truth)}")
        min_len = min(len(outputs), len(ground_truth))
        outputs = outputs[:min_len]
        ground_truth = ground_truth[:min_len]
    
    num_items = len(outputs)
    exact_matches = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_iter_count = 0.0
    
    incorrect_details = []
    detailed_results = []
    
    print("Evaluating answers...")
    print("-" * 80)
    
    for i, (output_item, gt_item) in enumerate(zip(outputs, ground_truth)):
        llm_answer = output_item.get("answer", "")
        gt_answer = gt_item.get("answer", "")
        query = output_item.get("query", gt_item.get("query", ""))
        out_iter = output_item.get("inference_steps", [])
        
        
        # Calculate metrics
        exact_match = calculate_exact_match(llm_answer, gt_answer)
        precision, recall, f1 = calculate_f1_score(llm_answer, gt_answer)
        iter_count = len(out_iter)
        
        # Update totals
        if exact_match:
            exact_matches += 1
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_iter_count += iter_count
        
        # Store detailed results
        result_detail = {
            "index": i,
            "query": query,
            "llm_answer": llm_answer,
            "ground_truth": gt_answer,
            "exact_match": exact_match,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "iter_count": round(iter_count, 4),
            "normalized_llm": normalize_answer(llm_answer),
            "normalized_gt": normalize_answer(gt_answer)
        }
        
        detailed_results.append(result_detail)
        
        # Collect incorrect answers for analysis
        if not exact_match:
            incorrect_details.append(result_detail)
            
        # Print progress for long evaluations
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_items} items...")
    
    # Calculate averages
    exact_match_accuracy = exact_matches / num_items
    avg_precision = total_precision / num_items
    avg_recall = total_recall / num_items
    avg_f1 = total_f1 / num_items
    avg_iter_count = total_iter_count / num_items
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Items Evaluated: {num_items}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({exact_matches}/{num_items})")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average iter count: {avg_iter_count:.4f}")
    print(f"Incorrect Answers: {len(incorrect_details)}")
    # Save detailed results if requested
    if detailed_output:
        # Save all detailed results
        detailed_file = "detailed_evaluation_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_items": num_items,
                    "exact_match_accuracy": exact_match_accuracy,
                    "exact_matches": exact_matches,
                    "average_precision": avg_precision,
                    "average_recall": avg_recall,
                    "average_f1_score": avg_f1,
                    "average_iter_count": avg_iter_count,
                    "incorrect_count": len(incorrect_details)
                },
                "detailed_results": detailed_results
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {detailed_file}")
        
        # Save incorrect answers for analysis
        if incorrect_details:
            wrong_ans_file = "wrong_answers_with_f1.json"
            with open(wrong_ans_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {
                        "total_incorrect": len(incorrect_details),
                        "percentage_incorrect": (len(incorrect_details) / num_items) * 100
                    },
                    "incorrect_answers": incorrect_details
                }, f, indent=2, ensure_ascii=False)
            print(f"Incorrect answers with F1 scores saved to: {wrong_ans_file}")
    
    return {
        "exact_match_accuracy": exact_match_accuracy,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1,
        "total_items": num_items,
        "average_iter_count": avg_iter_count,
        "exact_matches": exact_matches,
    }

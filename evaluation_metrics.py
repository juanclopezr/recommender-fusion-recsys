import numpy as np

#Precision at k
def calculate_custom_precision_at_k(true_items, predicted_items, k):
    """Calculates Precision at k."""
    total_precision = 0.0
    denominator = k
    if len(true_items) > 0 and len(predicted_items) >= k:
        for predicted_item in predicted_items[:k]:
            if predicted_item in true_items:
                total_precision += 1.0
    elif len(predicted_items) < k:
        # Fire exception
        raise ValueError("The length of predicted_items must be at least k.")
    if len(true_items) < k:
        denominator = len(true_items)
    return total_precision / denominator    


def calculate_precision_at_k(true_items, predicted_items, k):
    """Calculates Precision at k."""
    total_precision = 0.0
    if len(true_items) > 0 and len(predicted_items) >= k:
        for predicted_item in predicted_items:
            if predicted_item in true_items:
                total_precision += 1.0
    elif len(predicted_items) < k:
        # Fire exception
        raise ValueError("The length of predicted_items must be at least k.")
    return total_precision / k

def calculate_average_precision_at_k(true_items, predicted_items, k):
    """Calculates Average Precision at k."""
    total_ap = 0.0
    for true_item, predicted_list in zip(true_items, predicted_items):
        total_ap += calculate_precision_at_k(true_item, predicted_list[:k], k)
    return total_ap / len(true_items) if true_items else 0.0

def calculate_average_custom_precision_at_k(true_items, predicted_items, k):
    """Calculates Custom Average Precision at k."""
    total_ap = 0.0
    for true_item, predicted_list in zip(true_items, predicted_items):
        total_ap += calculate_custom_precision_at_k(true_item, predicted_list[:k], k)
    return total_ap / len(true_items) if true_items else 0.0

#Mean Reciprocal Rank

def calculate_mrr(true_items, predicted_items):
    """Calculates Mean Reciprocal Rank (MRR)."""
    index = 0
    total_rr = 0.0
    for predicted_list in predicted_items:
        index+=1
        if predicted_list in true_items:
            total_rr = 1.0 / index
    return total_rr if len(true_items) >= 1 else 0.0

def calculate_average_mrr(true_items, predicted_items):
    """Calculates Average Mean Reciprocal Rank (MRR)."""
    total_mrr = 0.0
    for true_item, predicted_list in zip(true_items, predicted_items):
        total_mrr += calculate_mrr(true_item, predicted_list)
    return total_mrr / len(true_items) if true_items else 0.0

#Normalized Discounted Cumulative Gain at k
import math

def dcg(relevances):
    """DCG using 2^rel - 1 discount."""
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))

def build_relevances(predicted, ground_truth):
    """
    Build relevance scores for each item in `predicted` using ground_truth order.
    """
    m = len(ground_truth)
    score_map = {item: m - idx for idx, item in enumerate(ground_truth)}  # top -> m
    return [score_map.get(item, 0) for item in predicted]


def ndcg(predicted, ground_truth, k=None):
    if k is None:
        k = len(predicted)
    pred_cut = predicted[:k]
    relevances = build_relevances(pred_cut, ground_truth)
    dcg_val = dcg(relevances)
    idcg_val = dcg(sorted(relevances, reverse=True))
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def calculate_average_ndcg_at_k(true_items, predicted_items, k):
    """Calculates Average NDCG at k."""
    total_ndcg = 0.0
    for true_item, predicted_list in zip(true_items, predicted_items):
        total_ndcg += ndcg(predicted_list, true_item, k)
    return total_ndcg / len(true_items) if true_items else 0.0

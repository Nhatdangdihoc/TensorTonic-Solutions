def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    relevant = set(relevant)

    hits = 0
    for item in top_k:
        if item in relevant:
            hits += 1

    precision = hits / k
    recall = hits / len(relevant)

    return [float(precision), float(recall)]
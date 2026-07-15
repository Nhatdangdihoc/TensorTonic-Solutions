import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.

    Args:
        y_true_list: list of arrays, binary relevance labels {0,1} per query
        y_score_list: list of arrays, real-valued scores per query (cùng độ dài)
        k: cutoff rank (optional). None -> dùng toàn bộ độ dài của từng query

    Returns:
        (map_value, ap_per_query): tuple(float, list[float])
    """
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)
        n = len(y_true)

        R = y_true.sum()  # tổng số item liên quan (toàn bộ list, KHÔNG cắt theo k)

        if R == 0:
            ap_per_query.append(0.0)
            continue

        # sắp xếp theo score giảm dần; stable để giữ thứ tự gốc khi bằng điểm
        order = np.argsort(-y_score, kind='stable')
        y_true_sorted = y_true[order]

        cutoff = n if k is None else min(k, n)
        y_true_top = y_true_sorted[:cutoff]

        ranks = np.arange(1, cutoff + 1)
        cum_rel = np.cumsum(y_true_top)          # số item liên quan tính đến rank i
        precision_at_i = cum_rel / ranks          # P(i), vector hoá

        ap = np.sum(precision_at_i * y_true_top) / R
        ap_per_query.append(float(ap))

    map_value = float(np.mean(ap_per_query)) if ap_per_query else 0.0
    return map_value, ap_per_query
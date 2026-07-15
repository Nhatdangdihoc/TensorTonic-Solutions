import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # ---- Trường hợp binary (chỉ quan tâm lớp pos_label) ----
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        return {"accuracy": accuracy, "precision": float(precision),
                "recall": float(recall), "f1": float(f1)}

    # ---- Các trường hợp multi-class: micro / macro / weighted ----
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precisions, recalls, f1s, supports = [], [], [], []
    tp_total = fp_total = fn_total = 0

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f = safe_div(2 * p * r, p + r)

        precisions.append(p); recalls.append(r); f1s.append(f); supports.append(support)
        tp_total += tp; fp_total += fp; fn_total += fn

    if average == "micro":
        precision = safe_div(tp_total, tp_total + fp_total)
        recall = safe_div(tp_total, tp_total + fn_total)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == "macro":
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

    elif average == "weighted":
        supports = np.array(supports, dtype=float)
        total = supports.sum()
        if total > 0:
            precision = np.sum(np.array(precisions) * supports) / total
            recall = np.sum(np.array(recalls) * supports) / total
            f1 = np.sum(np.array(f1s) * supports) / total
        else:
            precision = recall = f1 = 0.0

    else:
        raise ValueError(f"Unknown average type: {average}")

    return {"accuracy": accuracy, "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}
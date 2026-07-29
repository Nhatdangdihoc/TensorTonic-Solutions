def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_true) == 0:
        return 0.0

    true_positive = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(y_true, y_pred)
    )

    false_positive = len(y_true) - true_positive
    false_negative = len(y_true) - true_positive

    denominator = 2 * true_positive + false_positive + false_negative

    if denominator == 0:
        return 0.0

    return float((2 * true_positive) / denominator)
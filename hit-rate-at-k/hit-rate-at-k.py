def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Tính Hit Rate @ K (HR@K).

    HR@K = (số user có ít nhất 1 item đúng trong top-K gợi ý) / (tổng số user)

    Args:
        recommendations: list các list, mỗi phần tử là danh sách item được
                          gợi ý cho 1 user, đã sắp xếp theo độ ưu tiên giảm dần.
                          VD: [[item1, item2, ...], [item5, item3, ...], ...]
        ground_truth: list, mỗi phần tử tương ứng 1 user, có thể là:
                          - 1 item duy nhất user thực sự tương tác, hoặc
                          - 1 list/set nhiều item liên quan
        k: số lượng top item được xét (top-K)

    Returns:
        float: hit rate @ k, giá trị trong [0, 1]
    """
    if len(recommendations) != len(ground_truth):
        raise ValueError("recommendations và ground_truth phải có cùng số lượng user")

    if len(recommendations) == 0:
        return 0.0

    hits = 0
    for rec_items, true_items in zip(recommendations, ground_truth):
        top_k = set(rec_items[:k])

        # chuẩn hóa ground truth về set, phòng trường hợp chỉ có 1 item
        true_set = set(true_items) if isinstance(true_items, (list, set, tuple)) else {true_items}

        if top_k & true_set:  # có giao nhau -> tính là hit
            hits += 1

    return hits / len(recommendations)
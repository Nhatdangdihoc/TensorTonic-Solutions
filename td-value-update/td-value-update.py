import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    TD(0) value function update.

    V: dict hoặc array, ánh xạ trạng thái -> giá trị ước lượng V(s)
    s: trạng thái hiện tại
    r: phần thưởng nhận được sau khi thực hiện hành động tại s
    s_next: trạng thái kế tiếp
    alpha: tốc độ học (learning rate)
    gamma: hệ số chiết khấu (discount factor)

    Returns: V_new — hàm giá trị đã được cập nhật (không sửa V gốc, trả về bản mới)
    """
    # Tạo bản sao để không chỉnh sửa V gốc (in-place)
    V_new = V.copy()

    # TD target: ước lượng "tốt hơn" cho V(s), dựa trên phần thưởng thực tế
    # và giá trị ước lượng của trạng thái kế tiếp
    td_target = r + gamma * V_new[s_next]

    # TD error: chênh lệch giữa target và giá trị hiện tại
    td_error = td_target - V_new[s]

    # Cập nhật V(s) theo hướng giảm sai số, với bước nhảy alpha
    V_new[s] = V_new[s] + alpha * td_error

    return V_new
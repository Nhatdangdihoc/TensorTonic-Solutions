import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.

    Args:
        feature_size (int): size of the square feature grid (e.g. 19)
        image_size (int): size of the original square image (e.g. 600)
        scales (list of float): list of anchor scales (e.g. [64, 128, 256])
        aspect_ratios (list of float): list of aspect ratios (e.g. [0.5, 1.0, 2.0])

    Returns:
        list of [x1, y1, x2, y2]: anchor boxes in image coordinates
    """
    anchors = []

    # Step 1: stride
    stride = image_size / feature_size

    # Step 2: iterate grid cells in row-major order (i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            # Step 3: for each scale, then each aspect ratio
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)

                    # Step 4: convert to [x1, y1, x2, y2]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([x1, y1, x2, y2])

    return anchors


# --- Example usage ---
if __name__ == "__main__":
    anchors = generate_anchors(
        feature_size=3,
        image_size=600,
        scales=[100],
        aspect_ratios=[1.0]
    )
    for a in anchors:
        print(a)

    print(f"\nTotal anchors: {len(anchors)}")
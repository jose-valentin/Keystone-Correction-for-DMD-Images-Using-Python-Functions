import cv2
import numpy as np
import random

random.seed(42)  # same seed as compare_methods.py


# ---- Step 1: get the four corners of the shape ----
# approxPolyDP with 0.02 x perimeter epsilon first (same as compare_methods.py).
# If that does not return exactly 4 points, try a smaller epsilon. A nearly flat
# corner (close to 180 degrees) is dropped by the larger epsilon, which leaves 3 points.
def find_corners(image, epsilons=(0.02, 0.015, 0.01)):
    contours, trash = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(contour):
            contour = cnt
    perimeter = cv2.arcLength(contour, True)
    for eps in epsilons:
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        points = approx.reshape(-1, 2)
        if len(points) == 4:
            return points.astype("float32")
    return None  # no epsilon gave exactly 4 corners


# ---- Step 2: order the corners (cycle-preserving, from week 3) ----
# approxPolyDP already returns the points in a cycle around the shape.
# Only the winding direction and the starting corner are unknown.
def order_corners(points):
    # winding direction from the shoelace signed area
    area = 0.0
    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]
        area = area + (x1 * y2 - x2 * y1)
    ordered = []
    for i in range(4):
        ordered.append(points[i])
    if area < 0:
        ordered.reverse()
    # start the cycle at the corner nearest the top-left of the points' own bounding box
    min_x = ordered[0][0]
    min_y = ordered[0][1]
    for p in ordered:
        if p[0] < min_x:
            min_x = p[0]
        if p[1] < min_y:
            min_y = p[1]
    best_index = 0
    best_distance = None
    for i in range(4):
        distance = (ordered[i][0] - min_x) ** 2 + (ordered[i][1] - min_y) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = i
    result = []
    for i in range(4):
        result.append(ordered[(best_index + i) % 4])
    return np.float32(result)


# ---- Step 3: the correction function ----
# Four-point transform from PyImageSearch (2014). The output size rule (longer of the two
# widths, longer of the two heights) is ported directly from that article. The ordering and
# the target_size option are changes made here. target_size is (width, height) in pixels.
# Use it when the true size of the shape is known, because four corners alone cannot give it.
def correct_keystone(image, target_size=None):
    points = find_corners(image)
    if points is None:
        return None
    tl, tr, br, bl = order_corners(points)
    if target_size is None:
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)
        out_w = int(max(top, bottom))
        out_h = int(max(left, right))
    else:
        out_w, out_h = target_size
    source = np.float32([tl, tr, br, bl])
    target = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, transform, (out_w, out_h))


def extent_of(image):
    contours, trash = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(contour):
            contour = cnt
    trash, (w, h), trash = cv2.minAreaRect(contour)
    return cv2.contourArea(contour) / (w * h)


# ---- test harness: same 36 cases and generator as compare_methods.py ----
# Also returns the true corners, so the correction can be checked against ground truth.
def make_case(shape_type, magnitude, skew_amount):
    margin = magnitude // 6
    width = magnitude - 2 * margin
    if shape_type == "square":
        height = width
    else:
        height = int(width * 0.6)
    canvas_size = magnitude + (2 * skew_amount)
    base_x = margin + skew_amount
    base_y = margin + skew_amount
    image1 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.rectangle(image1, (base_x, base_y), (base_x + width, base_y + height), 255, -1)
    points_original = np.float32([
        [base_x, base_y], [base_x + width, base_y],
        [base_x + width, base_y + height], [base_x, base_y + height]
    ])
    corner0 = [base_x + random.randint(-skew_amount, skew_amount), base_y + random.randint(-skew_amount, skew_amount)]
    corner1 = [base_x + width + random.randint(-skew_amount, skew_amount), base_y + random.randint(-skew_amount, skew_amount)]
    corner2 = [base_x + width + random.randint(-skew_amount, skew_amount), base_y + height + random.randint(-skew_amount, skew_amount)]
    corner3 = [base_x + random.randint(-skew_amount, skew_amount), base_y + height + random.randint(-skew_amount, skew_amount)]
    points_skewed = np.float32([corner0, corner1, corner2, corner3])
    transform = cv2.getPerspectiveTransform(points_original, points_skewed)
    image2 = cv2.warpPerspective(image1, transform, (canvas_size, canvas_size))
    return image1, image2, points_original, points_skewed, width, height


# Mean distance between the detected corners and the true corners. The start corner of the
# detected cycle is unknown, so all four cyclic shifts are tried and the best one is kept.
# An odd shift means the detected "top" edge is really a side edge (width and height swapped).
def match_to_truth(ordered, truth):
    best_error = None
    best_shift = 0
    for shift in range(4):
        total_error = 0.0
        for i in range(4):
            total_error = total_error + np.linalg.norm(ordered[(i + shift) % 4] - truth[i])
        mean_error = total_error / 4.0
        if best_error is None or mean_error < best_error:
            best_error = mean_error
            best_shift = shift
    return best_error, best_shift


# Warp the distorted image back to the TRUE original rectangle using the detected corners,
# then measure how much it overlaps the undistorted image (1.0 = identical).
def overlap_with_original(image1, image2, ordered, shift, points_original):
    shifted = []
    for i in range(4):
        shifted.append(ordered[(i + shift) % 4])
    transform = cv2.getPerspectiveTransform(np.float32(shifted), points_original)
    back = cv2.warpPerspective(image2, transform, (image1.shape[1], image1.shape[0]))
    both = np.logical_and(image1 > 127, back > 127).sum()
    either = np.logical_or(image1 > 127, back > 127).sum()
    return both / either


if __name__ == "__main__":
    shape_types = ["square", "rectangle"]
    magnitudes = [150, 300, 600]
    skew_amounts = [0, 10, 20, 40, 60, 80]

    total = 0
    corrected_count = 0
    failed_count = 0
    distorted_count = 0
    gain_sum = 0.0
    error_sum = 0.0
    overlap_sum = 0.0
    ratio_sum = 0.0
    ratio_max = 0.0
    rectangle_count = 0
    swapped_count = 0
    sample_before = []
    sample_after = []

    for shape_type in shape_types:
        for magnitude in magnitudes:
            for skew_amount in skew_amounts:
                image1, image2, points_original, points_skewed, width, height = make_case(shape_type, magnitude, skew_amount)
                total = total + 1
                before = extent_of(image2)
                corrected = correct_keystone(image2)
                if corrected is None:
                    failed_count = failed_count + 1
                    print(shape_type, magnitude, skew_amount, "-> NOT CORRECTABLE (no epsilon gives 4 corners)")
                    continue
                after = extent_of(corrected)
                corrected_count = corrected_count + 1

                ordered = order_corners(find_corners(image2))
                corner_error, shift = match_to_truth(ordered, points_skewed)
                overlap = overlap_with_original(image1, image2, ordered, shift, points_original)
                out_h, out_w = corrected.shape[:2]
                true_ratio = width / height
                out_ratio = out_w / out_h
                ratio_error = abs(out_ratio - true_ratio) / true_ratio * 100

                print(shape_type, magnitude, skew_amount,
                      "-> extent before", round(before, 3), "after", round(after, 3),
                      " corner error", round(corner_error, 2), "px",
                      " overlap", round(overlap, 3),
                      " ratio error", round(ratio_error, 1), "%")

                if skew_amount > 0:
                    distorted_count = distorted_count + 1
                    gain_sum = gain_sum + (after - before)
                    error_sum = error_sum + corner_error
                    overlap_sum = overlap_sum + overlap
                    ratio_sum = ratio_sum + ratio_error
                    if ratio_error > ratio_max:
                        ratio_max = ratio_error
                    if shape_type == "rectangle":
                        rectangle_count = rectangle_count + 1
                        if shift % 2 == 1:
                            swapped_count = swapped_count + 1

                if shape_type == "square" and magnitude == 300:
                    sample_before.append(image2)
                    sample_after.append(corrected)

    print("")
    print("total cases:", total)
    print("corrected:", corrected_count, " not correctable:", failed_count)
    print("distorted cases corrected:", distorted_count)
    print("mean extent change (distorted only):", round(gain_sum / distorted_count, 4))
    print("mean corner error:", round(error_sum / distorted_count, 2), "px")
    print("mean overlap with original at true size:", round(overlap_sum / distorted_count, 3))
    print("mean aspect ratio error:", round(ratio_sum / distorted_count, 1), "%  max:", round(ratio_max, 1), "%")
    print("rectangles with width and height swapped:", swapped_count, "of", rectangle_count)

    # save a before/after strip for the square / 300 px series (saved next to this script)
    tiles = []
    for i in range(len(sample_before)):
        before_img = cv2.resize(sample_before[i], (200, 200))
        after_img = cv2.resize(sample_after[i], (200, 200))
        tiles.append(np.vstack([before_img, after_img]))
    cv2.imwrite("correction_samples.png", np.hstack(tiles))

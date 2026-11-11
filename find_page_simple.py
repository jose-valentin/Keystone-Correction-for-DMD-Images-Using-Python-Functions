import cv2
import numpy as np

from correct_keystone import order_corners

# Simplest page finder: try many brightness cutoffs on each color channel, collect every 4
# sided shape, and keep the one whose corners are closest to 90 degrees (the most square one).
#
# Source: OpenCV's squares.py sample,
# https://github.com/opencv/opencv/blob/4.x/samples/python/squares.py
#   - angle_cos is PORTED DIRECTLY.
#   - The loop follows the sample: split into B, G, R, sweep brightness cutoffs, findContours,
#     approxPolyDP at 0.02 of the perimeter, keep shapes with 4 corners and a minimum area.
#   - ADAPTED WITH REAL CHANGES: no Canny and no blur; the cutoffs are 50 to 240 in steps of
#     10; the sample keeps every shape with max_cos < 0.1, but a page seen at an angle is not
#     that square, so this ranks the shapes and keeps the best one; the minimum area is a
#     fraction of the photo (5%) instead of a fixed 1000 pixels.
#   - ADDED (not in the sample): shapes touching the photo edge are skipped, because they
#     are the table and not the page.
# Same idea as the Week 2 angle-cosine detector, applied to a real photo.

MIN_AREA_FRACTION = 0.05
BORDER_MARGIN = 3        # shapes touching the photo edge are the table, not the page


# Ported directly from squares.py. Cosine of the angle at p1, between p1 to p0 and p1 to p2.
# 0 means a right angle.
def angle_cos(p0, p1, p2):
    d1 = (p0 - p1).astype("float")
    d2 = (p2 - p1).astype("float")
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# Returns (corners, max_cos). corners is 4x2 ordered top-left, top-right, bottom-right,
# bottom-left, or None. Lower max_cos means more square.
def find_page_simple(image):
    height, width = image.shape[:2]
    min_area = MIN_AREA_FRACTION * height * width

    best_quad = None
    best_cos = 2.0
    for channel in cv2.split(image):
        for threshold in range(50, 250, 10):
            trash, binary = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
            contours, trash = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                length = cv2.arcLength(contour, True)
                shape = cv2.approxPolyDP(contour, 0.02 * length, True)
                if len(shape) != 4:
                    continue
                box_x, box_y, box_w, box_h = cv2.boundingRect(shape)
                if box_x <= BORDER_MARGIN or box_y <= BORDER_MARGIN:
                    continue
                if box_x + box_w >= width - BORDER_MARGIN or box_y + box_h >= height - BORDER_MARGIN:
                    continue
                if cv2.contourArea(shape) <= min_area:
                    continue
                points = shape.reshape(-1, 2)
                max_cos = 0.0
                for i in range(4):
                    value = angle_cos(points[i], points[(i + 1) % 4], points[(i + 2) % 4])
                    if value > max_cos:
                        max_cos = value
                if max_cos < best_cos:
                    best_cos = max_cos
                    best_quad = points.astype("float32")

    if best_quad is None:
        return None, 1.0
    return order_corners(best_quad), best_cos


if __name__ == "__main__":
    import sys

    photo = cv2.imread(sys.argv[1])
    corners, max_cos = find_page_simple(photo)
    print("max_cos:", max_cos)
    print(corners)
    if corners is not None:
        points = np.int32(corners).reshape(-1, 1, 2)
        cv2.polylines(photo, [points], True, (0, 220, 0), 5)
        cv2.imwrite("find_page_simple_result.png", cv2.resize(photo, (960, 540)))

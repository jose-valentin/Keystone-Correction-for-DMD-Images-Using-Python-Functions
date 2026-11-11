import sys

import cv2
import numpy as np

from find_page_simple import find_page_simple

# Demo: one photo in, a picture out with the found corners and the straightened page.
#   python3 demo_simple.py <photo> [output picture]
#
# Step 1, find the four corners: find_page_simple.py (adapted from OpenCV's squares.py).
# Step 2, straighten the page: the same steps as correct_keystone.py from Week 4
# (getPerspectiveTransform and warpPerspective, output size rule from PyImageSearch's
# four point transform, https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/).
# The size comes from the corners alone, so the width to height ratio can be off. That is the
# Week 5 problem that estimate_aspect_ratio is meant to fix.


# Same steps as correct_keystone() in correct_keystone.py, but the corners are given instead
# of being found inside the function. corners are ordered top-left, top-right, bottom-right,
# bottom-left.
def straighten(photo, corners):
    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    out_w = int(max(top, bottom))
    out_h = int(max(left, right))
    source = np.float32([tl, tr, br, bl])
    target = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(photo, transform, (out_w, out_h))


def resize_to_height(image, height):
    ratio = height / float(image.shape[0])
    return cv2.resize(image, (max(1, int(image.shape[1] * ratio)), height))


def run_demo(photo_path, output_path):
    photo = cv2.imread(photo_path)
    if photo is None:
        print("could not read", photo_path)
        return False
    corners, max_cos = find_page_simple(photo)
    if corners is None:
        print("no page found")
        return False
    print("max_cos (0 = perfect right angles):", round(float(max_cos), 3))
    print("corners (top-left, top-right, bottom-right, bottom-left):")
    for corner in corners:
        print("  ", round(float(corner[0]), 1), round(float(corner[1]), 1))

    outlined = photo.copy()
    points = np.int32(corners).reshape(-1, 1, 2)
    cv2.polylines(outlined, [points], True, (0, 220, 0), 8)
    straightened = straighten(photo, corners)
    print("straightened size:", straightened.shape[1], "by", straightened.shape[0])

    left = resize_to_height(outlined, 540)
    right = resize_to_height(straightened, 540)
    gap = np.zeros((540, 20, 3), dtype=np.uint8)
    cv2.imwrite(output_path, np.hstack([left, gap, right]))
    print("saved", output_path)
    return True


if __name__ == "__main__":
    output = "demo_result.png"
    if len(sys.argv) > 2:
        output = sys.argv[2]
    run_demo(sys.argv[1], output)

import cv2
import numpy as np

from correct_keystone import order_corners

# Week 6: find the page (or projected image) in a photo without one global threshold.
#
# Source and what is changed:
# The threshold sweep idea comes from OpenCV's squares.py sample
# (https://github.com/opencv/opencv/blob/4.x/samples/python/squares.py). That sample tries
# several thresholds and keeps convex 4 sided contours. Adapted with real changes here:
#   - the sweep is over plain brightness thresholds only (no Canny, no color planes)
#   - a morphological opening breaks thin bridges (a cable touching the page)
#   - candidates are ranked by a score instead of an area and angle test, because a
#     page seen in perspective has no right angles to test
# The score and the edge check are written for this project, not taken from a source.

EDGE_OFFSET = 4          # pixels inside and outside the quad edge compared (on the small image)
EDGE_DIFFERENCE = 15     # gray levels the inside must be brighter than the outside
SAMPLES_PER_EDGE = 20
MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 0.90


# Four corners from one contour. The convex hull is used so a bump in the outline
# (a cable, a finger) does not add extra corners. Smallest epsilon that gives 4 points wins.
def quad_from_contour(contour):
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    epsilons = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)
    for eps in epsilons:
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype("float32")
    # a ragged outline can jump from 5 points straight to 3, so ask for exactly 4 directly
    hull_float = hull.astype("float32")
    four = cv2.approxPolyN(hull_float, 4, epsilon_percentage=-1.0, ensure_convex=True)
    if four is not None and len(four.reshape(-1, 2)) == 4:
        return four.reshape(-1, 2).astype("float32")
    return None


# Fraction of points along the four edges where the inside is clearly brighter than
# the outside. A real page border scores near 1.0. A quad drawn across a blob of table
# scores low, because most of its edge has no brightness step.
def edge_support(gray, quad):
    height, width = gray.shape
    center_x = float(quad[:, 0].mean())
    center_y = float(quad[:, 1].mean())
    strong = 0
    total = 0
    for i in range(4):
        p = quad[i]
        q = quad[(i + 1) % 4]
        dx = float(q[0] - p[0])
        dy = float(q[1] - p[1])
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return 0.0
        nx = dy / length
        ny = -dx / length
        mid_x = float(p[0] + q[0]) / 2.0
        mid_y = float(p[1] + q[1]) / 2.0
        # make the normal point away from the center of the quad
        here = (mid_x - center_x) ** 2 + (mid_y - center_y) ** 2
        moved = (mid_x + nx - center_x) ** 2 + (mid_y + ny - center_y) ** 2
        if moved < here:
            nx = -nx
            ny = -ny
        for k in range(SAMPLES_PER_EDGE):
            t = (k + 0.5) / SAMPLES_PER_EDGE
            x = float(p[0]) + t * dx
            y = float(p[1]) + t * dy
            inside_x = int(round(x - nx * EDGE_OFFSET))
            inside_y = int(round(y - ny * EDGE_OFFSET))
            outside_x = int(round(x + nx * EDGE_OFFSET))
            outside_y = int(round(y + ny * EDGE_OFFSET))
            total = total + 1
            inside_ok = inside_x >= 0 and inside_x < width and inside_y >= 0 and inside_y < height
            outside_ok = outside_x >= 0 and outside_x < width and outside_y >= 0 and outside_y < height
            if inside_ok and outside_ok:
                difference = int(gray[inside_y, inside_x]) - int(gray[outside_y, outside_x])
                if difference > EDGE_DIFFERENCE:
                    strong = strong + 1
    return strong / float(total)


# Score of one candidate quad: how well the quad describes the blob (fit) times
# how much of its border is a real brightness edge (edge support). Both are 0 to 1.
def score_quad(gray, hull_area, quad):
    quad_area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if quad_area <= 0 or hull_area <= 0:
        return 0.0
    fit = min(hull_area, quad_area) / max(hull_area, quad_area)
    support = edge_support(gray, quad)
    return fit * support


# Intersection of two lines given as (vx, vy, x0, y0). Returns None if they are parallel.
def intersect_lines(line_a, line_b):
    vxa, vya, x0a, y0a = line_a
    vxb, vyb, x0b, y0b = line_b
    det = -vxa * vyb + vxb * vya
    if abs(det) < 0.000001:
        return None
    rx = x0b - x0a
    ry = y0b - y0a
    t = (-rx * vyb + vxb * ry) / det
    return [x0a + t * vxa, y0a + t * vya]


# Week 6 refinement. The threshold boundary found on the small image sits a few pixels off the
# real page edge. For each edge, look along the outward normal at full resolution for the
# strongest bright-to-dark step, fit a line through those points, then intersect neighboring
# lines to get the corners. Written for this project. If any edge has too few good points the
# original corners are returned unchanged.
def refine_corners(image, corners, search=20, positions=24):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    height, width = gray.shape
    center_x = float(corners[:, 0].mean())
    center_y = float(corners[:, 1].mean())
    lines = []
    for i in range(4):
        p = corners[i]
        q = corners[(i + 1) % 4]
        dx = float(q[0] - p[0])
        dy = float(q[1] - p[1])
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return corners
        nx = dy / length
        ny = -dx / length
        mid_x = float(p[0] + q[0]) / 2.0
        mid_y = float(p[1] + q[1]) / 2.0
        here = (mid_x - center_x) ** 2 + (mid_y - center_y) ** 2
        moved = (mid_x + nx - center_x) ** 2 + (mid_y + ny - center_y) ** 2
        if moved < here:
            nx = -nx
            ny = -ny
        edge_points = []
        for k in range(positions):
            t = 0.1 + 0.8 * (k + 0.5) / positions
            x = float(p[0]) + t * dx
            y = float(p[1]) + t * dy
            best_drop = 0
            best_offset = None
            for offset in range(-search, search + 1):
                before_x = int(round(x + nx * (offset - 2)))
                before_y = int(round(y + ny * (offset - 2)))
                after_x = int(round(x + nx * (offset + 2)))
                after_y = int(round(y + ny * (offset + 2)))
                if before_x < 0 or before_x >= width or before_y < 0 or before_y >= height:
                    continue
                if after_x < 0 or after_x >= width or after_y < 0 or after_y >= height:
                    continue
                drop = int(gray[before_y, before_x]) - int(gray[after_y, after_x])
                if drop > best_drop:
                    best_drop = drop
                    best_offset = offset
            if best_offset is not None and best_drop >= EDGE_DIFFERENCE:
                edge_points.append([x + nx * best_offset, y + ny * best_offset])
        if len(edge_points) < 8:
            return corners
        points = np.float32(edge_points).reshape(-1, 1, 2)
        fitted = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lines.append((float(fitted[0][0]), float(fitted[1][0]), float(fitted[2][0]), float(fitted[3][0])))
    refined = []
    for i in range(4):
        point = intersect_lines(lines[(i - 1) % 4], lines[i])
        if point is None:
            return corners
        refined.append(point)
    return np.float32(refined)


# Returns (corners, score). corners is a 4x2 float32 array ordered top-left, top-right,
# bottom-right, bottom-left in the original image size, or None if no candidate was found.
def find_page_corners_with_score(image, max_side=640, refine=True):
    height, width = image.shape[:2]
    scale = max_side / float(max(height, width))
    if scale > 1.0:
        scale = 1.0
    small = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    if len(small.shape) == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    small_area = float(gray.shape[0] * gray.shape[1])
    kernel = np.ones((5, 5), np.uint8)

    best_score = 0.0
    best_quad = None
    for threshold in range(50, 250, 10):
        trash, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, trash = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA_FRACTION * small_area or area > MAX_AREA_FRACTION * small_area:
                continue
            quad = quad_from_contour(contour)
            if quad is None:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            score = score_quad(gray, hull_area, quad)
            if score > best_score:
                best_score = score
                best_quad = quad

    if best_quad is None:
        return None, 0.0
    ordered = order_corners(best_quad / scale)
    if refine:
        ordered = refine_corners(image, ordered)
    return ordered, best_score


def find_page_corners(image, max_side=640, refine=True):
    corners, trash = find_page_corners_with_score(image, max_side, refine)
    return corners


# Baseline for comparison: one global Otsu threshold, largest blob, four corners.
# Modeled on the week 5 segmentation (blur, Otsu, largest contour). Not the exact week 5 code.
def otsu_baseline_corners(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    trash, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, trash = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(contour):
            contour = cnt
    perimeter = cv2.arcLength(contour, True)
    for eps in (0.02, 0.015, 0.01):
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(-1, 2).astype("float32"))
    return None

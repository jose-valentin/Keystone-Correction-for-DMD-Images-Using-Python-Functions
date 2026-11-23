import cv2
import numpy as np

from find_page_simple import find_page_simple

# The functions behind the dashboard. No window code in here, so it can be tested alone.
#
# Step 1, finding the corners: find_page_simple.py (adapted from OpenCV's squares.py).
# Step 2, straightening: the same steps as straighten() in demo_simple.py and
# correct_keystone() in correct_keystone.py (getPerspectiveTransform and warpPerspective,
# output size rule from PyImageSearch's four point transform,
# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/).
# The size comes from the corners alone, so the width to height ratio can be off.
# Everything else in this file (loading, outline drawing, the result object) is written
# for this project.

# A result with max_cos above this is flagged as low confidence. This number is a first guess
# from four laptop photos (the two good ones scored 0.11 and 0.14, the two wrong shapes 0.22
# and 0.31). It is NOT tuned. A page seen at a steep angle can score higher and still be right.
LOW_CONFIDENCE_COS = 0.2

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]
CORNER_LABELS = ["TL", "TR", "BR", "BL"]


class AnalysisResult:
    def __init__(self):
        self.found = False
        self.corners = None          # 4x2 array: top-left, top-right, bottom-right, bottom-left
        self.max_cos = 1.0
        self.low_confidence = False
        self.photo_width = 0
        self.photo_height = 0
        self.outline_image = None    # the photo with the found outline drawn on it
        self.straightened = None     # the straightened page, or None
        self.status_text = ""


# Read a picture from disk. cv2.imread can fail on Windows paths with accents (for example
# a user folder named "Jose" with an accent), so the bytes are read with numpy instead.
def load_photo(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# Save a picture to disk. Same reason as load_photo. Returns True if it was written.
def save_image(path, image):
    extension = ".png"
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        extension = ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True


# Same steps as straighten() in demo_simple.py. corners are ordered top-left, top-right,
# bottom-right, bottom-left.
def straighten(photo, corners):
    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    out_w = int(max(top, bottom))
    out_h = int(max(left, right))
    if out_w < 2 or out_h < 2:
        return None
    source = np.float32([tl, tr, br, bl])
    target = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(photo, transform, (out_w, out_h))


# Draw the outline and the four corner labels on a copy of the photo. Line and text sizes
# follow the photo size so they stay visible on big photos.
def draw_outline(photo, corners):
    picture = photo.copy()
    height, width = picture.shape[:2]
    thickness = max(2, int(max(height, width) / 250))
    font_scale = max(0.6, max(height, width) / 1200.0)
    points = np.int32(corners).reshape(-1, 1, 2)
    cv2.polylines(picture, [points], True, (0, 200, 0), thickness)
    for i in range(4):
        x = int(corners[i][0])
        y = int(corners[i][1])
        cv2.circle(picture, (x, y), thickness * 3, (0, 0, 255), -1)
        cv2.putText(picture, CORNER_LABELS[i], (x + thickness * 4, y - thickness * 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    return picture


# The whole analysis for one photo: find the corners, draw the outline, straighten.
def analyze_photo(photo):
    result = AnalysisResult()
    result.photo_height, result.photo_width = photo.shape[:2]
    result.outline_image = photo.copy()

    corners, max_cos = find_page_simple(photo)
    result.max_cos = float(max_cos)
    if corners is None:
        result.status_text = "No page found."
        return result

    result.found = True
    result.corners = corners
    result.outline_image = draw_outline(photo, corners)
    result.straightened = straighten(photo, corners)
    if result.max_cos > LOW_CONFIDENCE_COS:
        result.low_confidence = True
        result.status_text = "Page found, but low confidence. Check the outline."
    else:
        result.status_text = "Page found."
    return result


# Shrink a picture to fit inside max_width by max_height (never enlarge it) and return PNG
# bytes, which Tkinter can show without any extra library.
def to_png_bytes(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", small)
    if not ok:
        return None
    return encoded.tobytes()


# The results as plain text lines for the text box.
def result_lines(result, file_name):
    lines = []
    lines.append("File: " + file_name)
    lines.append("Photo size: %d by %d px" % (result.photo_width, result.photo_height))
    lines.append("Status: " + result.status_text)
    if not result.found:
        lines.append("Tip: use the white calibration frame and keep the whole screen in view.")
        return lines
    lines.append("Confidence: max_cos %.3f (0 is perfect right angles, lower is better)" % result.max_cos)
    lines.append("")
    lines.append("Corners (x, y in photo pixels):")
    for i in range(4):
        lines.append("  %-13s %8.1f %8.1f" % (CORNER_NAMES[i], result.corners[i][0], result.corners[i][1]))
    lines.append("")
    if result.straightened is not None:
        straight_height, straight_width = result.straightened.shape[:2]
        lines.append("Straightened size: %d by %d px" % (straight_width, straight_height))
        lines.append("The size comes from the corners alone, so the width to height ratio can be off.")
    return lines

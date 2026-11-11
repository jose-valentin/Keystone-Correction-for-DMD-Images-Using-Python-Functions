import cv2, numpy as np
from find_page_simple import find_page_simple
FW, FH = 1280, 720            # projector (DMD) frame size
CW, CH = 1920, 1080           # camera image size

# source picture to show: a grid with a border and an arrow so tilt is easy to see
src = np.full((FH, FW, 3), 235, np.uint8)
for x in range(0, FW, 80): cv2.line(src, (x, 0), (x, FH), (120, 120, 120), 1)
for y in range(0, FH, 80): cv2.line(src, (0, y), (FW, y), (120, 120, 120), 1)
cv2.rectangle(src, (0, 0), (FW - 1, FH - 1), (0, 0, 200), 12)
cv2.putText(src, "SOURCE IMAGE", (330, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (200, 0, 0), 6)

frame_corners = np.float32([[0, 0], [FW, 0], [FW, FH], [0, FH]])
# the keystone distortion: where the frame lands in the camera image (unknown to the program)
quad_true = np.float32([[430, 190], [1500, 120], [1330, 930], [560, 800]])
D = cv2.getPerspectiveTransform(frame_corners, quad_true)

# step 1: project a plain white frame, photograph it
white = np.full((FH, FW, 3), 255, np.uint8)
photo = np.zeros((CH, CW, 3), np.uint8); photo[:] = (40, 45, 50)
photo = np.where(cv2.warpPerspective(white, D, (CW, CH)) > 0, cv2.warpPerspective(white, D, (CW, CH)), photo)
# step 2: find the four corners in the photo
quad, trash = find_page_simple(photo.astype(np.uint8))
print("true corners :", quad_true.round().tolist())
print("found corners:", quad.round().tolist())
print("corner error px:", np.linalg.norm(quad - quad_true, axis=1).round(1).tolist())
# step 3: camera -> frame mapping (this is the same transform used to straighten a photo)
H = cv2.getPerspectiveTransform(np.float32(quad), frame_corners)
# step 4: choose where the corrected picture should appear: a 16:9 rectangle inside the found quad
cx, cy = quad.mean(axis=0)
best = None
for w in range(1000, 100, -2):
    h = w * 9 / 16
    rect = np.float32([[cx - w/2, cy - h/2], [cx + w/2, cy - h/2], [cx + w/2, cy + h/2], [cx - w/2, cy + h/2]])
    inside = all(cv2.pointPolygonTest(quad.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0 for p in rect)
    if inside:
        best = rect; break
R = best
# step 5: where those rectangle corners sit in projector-frame coordinates
Q = cv2.perspectiveTransform(R.reshape(-1, 1, 2), H).reshape(-1, 2)
# step 6: pre-warp the source so its corners land on Q, then show the pre-warped frame
P = cv2.getPerspectiveTransform(frame_corners, np.float32(Q))
prewarped = cv2.warpPerspective(src, P, (FW, FH))
# step 7: what the camera sees now
seen = np.zeros((CH, CW, 3), np.uint8); seen[:] = (40, 45, 50)
w2 = cv2.warpPerspective(prewarped, D, (CW, CH))
mask = cv2.warpPerspective(np.full((FH, FW), 255, np.uint8), D, (CW, CH)) > 0
seen[mask] = w2[mask]
# measure: where do the source corners end up in the camera image vs the wanted rectangle R
ends = cv2.perspectiveTransform(Q.reshape(-1, 1, 2), D).reshape(-1, 2)
print("wanted rectangle:", R.round().tolist())
print("where corners land:", ends.round().tolist())
print("error px:", np.linalg.norm(ends - R, axis=1).round(1).tolist(), "of a", int(R[1][0] - R[0][0]), "px wide picture")
def tile(img, text):
    t = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
    t = cv2.copyMakeBorder(t, 40, 6, 6, 6, cv2.BORDER_CONSTANT, value=(13, 13, 13))
    cv2.putText(t, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (242, 242, 239), 2, cv2.LINE_AA)
    return t
unwarped = cv2.warpPerspective(src, D, (CW, CH)); naive = photo.copy(); naive[mask] = unwarped[mask]
a = tile(naive, "1. Projected as is: crooked")
b = tile(cv2.resize(prewarped, (1920, 1080)) if False else np.pad(prewarped, ((0,0),(0,0),(0,0))), "2. Pre-warped frame sent to DMD")
c = tile(seen, "3. What the camera sees now")
h = max(x.shape[0] for x in (a, b, c))
def padh(x): return cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(13, 13, 13))
cv2.imwrite("projector_sim.png", np.hstack([padh(a), padh(b), padh(c)]))

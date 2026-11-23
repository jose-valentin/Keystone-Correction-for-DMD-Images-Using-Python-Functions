import sys

import cv2
import numpy as np

# Standard calibration frame for the DMD projector (1920 by 1080, the DLP6500 size).
#
# Plain white over the whole frame, so the finder sees one bright rectangle whose outline is
# the edge of the DMD image. Four colored squares sit INSIDE the white area, away from the
# edges, so they never change the outline. They tell the four corners apart even if the photo
# is rotated:
#   top-left red, top-right green, bottom-right blue, bottom-left black.
#
# Written for this project. Not taken from a source.

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MARKER_SIZE = 120        # side of each colored square, in pixels
MARKER_MARGIN = 60       # distance from the frame edge to the square, in pixels


def make_calibration_image(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    near = MARKER_MARGIN
    far_x = width - MARKER_MARGIN - MARKER_SIZE
    far_y = height - MARKER_MARGIN - MARKER_SIZE
    # colors are in BGR order, as OpenCV uses
    red = (0, 0, 255)
    green = (0, 200, 0)
    blue = (255, 0, 0)
    black = (0, 0, 0)
    cv2.rectangle(image, (near, near), (near + MARKER_SIZE, near + MARKER_SIZE), red, -1)
    cv2.rectangle(image, (far_x, near), (far_x + MARKER_SIZE, near + MARKER_SIZE), green, -1)
    cv2.rectangle(image, (far_x, far_y), (far_x + MARKER_SIZE, far_y + MARKER_SIZE), blue, -1)
    cv2.rectangle(image, (near, far_y), (near + MARKER_SIZE, far_y + MARKER_SIZE), black, -1)
    return image


if __name__ == "__main__":
    output = "calibration_frame.png"
    if len(sys.argv) > 1:
        output = sys.argv[1]
    cv2.imwrite(output, make_calibration_image())
    print("saved", output)

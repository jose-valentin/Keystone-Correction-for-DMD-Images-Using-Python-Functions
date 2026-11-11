import csv
import gzip
import os
import sys

import cv2
import numpy as np

from correct_keystone import match_to_truth
from demo_simple import straighten
from find_page_simple import angle_cos, find_page_simple, MIN_AREA_FRACTION, BORDER_MARGIN

# Picture helpers for the demo. NOT part of the page finder. They repeat the finder's steps
# so each one can be drawn, and they check that the drawn winner is the same one
# find_page_simple() returns.
#   python3 make_figures.py process <photo> <output png>
#   python3 make_figures.py gallery <SmartDoc folder> <output png>

TILE_W = 480
TILE_H = 270
CHANNEL_NAMES = ("blue", "green", "red")


def to_tile(image, text):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(image, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (TILE_W, 26), (0, 0, 0), -1)
    cv2.putText(tile, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return tile


# Every 4 sided shape the finder keeps, with its channel, cutoff and max_cos.
# Same filters as find_page_simple().
def all_candidates(image):
    height, width = image.shape[:2]
    min_area = MIN_AREA_FRACTION * height * width
    channels = cv2.split(image)
    found = []
    for channel_index in range(3):
        for threshold in range(50, 250, 10):
            trash, binary = cv2.threshold(channels[channel_index], threshold, 255, cv2.THRESH_BINARY)
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
                found.append((max_cos, channel_index, threshold, contour, shape))
    return found


def make_process_figure(photo_path, output_path):
    photo = cv2.imread(photo_path)
    candidates = all_candidates(photo)
    best = candidates[0]
    for item in candidates:
        if item[0] < best[0]:
            best = item
    best_cos, best_channel, best_threshold, best_contour, best_shape = best

    # check: the drawn winner is the same page the finder returns
    corners, trash = find_page_simple(photo)
    check_error, trash = match_to_truth(corners, np.float32(best_shape.reshape(-1, 2)))
    print("drawn winner matches find_page_simple, difference in px:", round(float(check_error), 3))

    channels = cv2.split(photo)
    tiles = []
    tiles.append(to_tile(photo, "1. Original photo"))
    tiles.append(to_tile(channels[0], "2. Blue channel: page bright, table dark"))
    tiles.append(to_tile(channels[1], "3. Green channel"))
    tiles.append(to_tile(channels[2], "4. Red channel"))

    trash, mask = cv2.threshold(channels[best_channel], best_threshold, 255, cv2.THRESH_BINARY)
    tiles.append(to_tile(mask, "5. Cutoff %d on %s: white = brighter" % (best_threshold, CHANNEL_NAMES[best_channel])))

    contours, trash = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    outlines = photo.copy()
    cv2.drawContours(outlines, contours, -1, (0, 200, 255), 4)
    tiles.append(to_tile(outlines, "6. Outline of every white blob"))

    kept = photo.copy()
    for item in candidates:
        cv2.polylines(kept, [item[4]], True, (255, 160, 0), 4)
    tiles.append(to_tile(kept, "7. Shapes with 4 corners kept (%d)" % len(candidates)))

    winner = photo.copy()
    cv2.polylines(winner, [best_shape], True, (0, 220, 0), 8)
    tiles.append(to_tile(winner, "8. Most square wins (max_cos %.2f)" % best_cos))

    straightened = straighten(photo, corners)
    canvas = np.zeros((photo.shape[0], photo.shape[1], 3), dtype=np.uint8)
    ratio = photo.shape[0] / float(straightened.shape[0])
    resized = cv2.resize(straightened, (max(1, int(straightened.shape[1] * ratio)), photo.shape[0]))
    fit = min(resized.shape[1], canvas.shape[1])
    canvas[:, 0:fit] = resized[:, 0:fit]
    tiles.append(to_tile(canvas, "9. Straightened with the Week 4 steps"))

    rows = []
    for r in range(3):
        rows.append(np.hstack(tiles[r * 3:(r + 1) * 3]))
    cv2.imwrite(output_path, np.vstack(rows))
    print("saved", output_path)


# ---- successes and failures gallery ----
def true_corners(row):
    return np.float32([
        [float(row["tl_x"]), float(row["tl_y"])],
        [float(row["tr_x"]), float(row["tr_y"])],
        [float(row["br_x"]), float(row["br_y"])],
        [float(row["bl_x"]), float(row["bl_y"])],
    ])


def gallery_tile(photo, truth, found, text):
    picture = photo.copy()
    cv2.polylines(picture, [np.int32(truth).reshape(-1, 1, 2)], True, (0, 0, 255), 8)
    if found is not None:
        cv2.polylines(picture, [np.int32(found).reshape(-1, 1, 2)], True, (0, 220, 0), 8)
    return to_tile(picture, text)


def make_gallery(dataset_dir, output_path):
    handle = gzip.open(os.path.join(dataset_dir, "metadata.csv.gz"), "rt")
    rows = []
    for row in csv.DictReader(handle):
        if int(row["frame_index"]) % 30 == 0:
            rows.append(row)
    handle.close()

    wanted = ["ok background01", "ok background02", "ok background03",
              "near miss", "wrong object", "no page background05"]
    chosen = {}
    for row in rows:
        if len(chosen) == len(wanted):
            break
        photo = cv2.imread(os.path.join(dataset_dir, row["image_path"]))
        truth = true_corners(row)
        found, trash = find_page_simple(photo)
        error = None
        if found is not None:
            error, trash = match_to_truth(found, truth)
        background = row["bg_name"]
        kind = None
        if error is not None and error <= 5.0 and background in ("background01", "background02", "background03"):
            kind = "ok " + background
        elif error is not None and 12.0 <= error <= 30.0:
            kind = "near miss"
        elif error is not None and error > 150.0:
            kind = "wrong object"
        elif found is None and background == "background05":
            kind = "no page background05"
        if kind is not None and kind not in chosen:
            chosen[kind] = (photo, truth, found, error, background)

    tiles = []
    for kind in wanted:
        if kind not in chosen:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
            continue
        photo, truth, found, error, background = chosen[kind]
        if kind.startswith("ok"):
            text = "right, %s, error %.0f px" % (background[-2:], error)
        elif kind == "near miss":
            text = "close but off, %s, error %.0f px" % (background[-2:], error)
        elif kind == "wrong object":
            text = "wrong shape, %s, error %.0f px" % (background[-2:], error)
        else:
            text = "no page found, %s (cluttered)" % background[-2:]
        tiles.append(gallery_tile(photo, truth, found, text))
        print(kind, background, "error", None if error is None else round(float(error), 1))
    top = np.hstack(tiles[0:3])
    bottom = np.hstack(tiles[3:6])
    cv2.imwrite(output_path, np.vstack([top, bottom]))
    print("saved", output_path, "(red = true corners, green = found)")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "process":
        make_process_figure(sys.argv[2], sys.argv[3])
    if mode == "gallery":
        make_gallery(sys.argv[2], sys.argv[3])

import csv
import gzip
import os
import sys

import cv2
import numpy as np

# The folder this script lives in holds correct_keystone.py and find_page.py. It is added to
# the path by hand because the script is run with python -I (isolated mode) when it reads
# downloaded data.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correct_keystone import match_to_truth
from find_page import find_page_corners_with_score, otsu_baseline_corners

# Runs the week 6 page finder on real SmartDoc 2015 Challenge 1 photos and compares the
# corners with the dataset's true corners. Usage:
#   python3 -I evaluate_smartdoc.py <dataset folder> [step]
# step = 10 uses frames numbered 10, 20, 30 and so on within each video, the same subset as
# the week 5 full chain test (2,421 of the 24,889 images).

SUCCESS_PIXELS = 10.0     # same limit as week 5 (within 10 px of the true corners)
CONFIDENT = 0.8           # score at or above this counts as confident


def read_rows(dataset_dir, step):
    path = os.path.join(dataset_dir, "metadata.csv.gz")
    rows = []
    handle = gzip.open(path, "rt")
    reader = csv.DictReader(handle)
    for row in reader:
        frame_index = int(row["frame_index"])
        if frame_index % step == 0:
            rows.append(row)
    handle.close()
    return rows


# The csv lists the corners as tl, bl, br, tr. The finder returns tl, tr, br, bl.
def true_corners(row):
    return np.float32([
        [float(row["tl_x"]), float(row["tl_y"])],
        [float(row["tr_x"]), float(row["tr_y"])],
        [float(row["br_x"]), float(row["br_y"])],
        [float(row["bl_x"]), float(row["bl_y"])],
    ])


# Mean and largest corner distance, after trying all four starting corners.
def corner_errors(found, truth):
    mean_error, shift = match_to_truth(found, truth)
    largest = 0.0
    for i in range(4):
        distance = float(np.linalg.norm(found[(i + shift) % 4] - truth[i]))
        if distance > largest:
            largest = distance
    return float(mean_error), largest


def draw_quad(image, quad, color, thickness):
    points = np.int32(quad).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, thickness)


def median_of(values):
    if len(values) == 0:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    step = 10
    if len(sys.argv) > 2:
        step = int(sys.argv[2])

    rows = read_rows(dataset_dir, step)
    print("images to test:", len(rows))

    backgrounds = ["background01", "background02", "background03", "background04", "background05"]
    stats = {}
    for name in backgrounds:
        stats[name] = {"images": 0, "otsu_found": 0, "otsu_mean_ok": 0, "found": 0,
                       "mean_ok": 0, "max_ok": 0, "errors": [],
                       "confident": 0, "confident_ok": 0, "unsure": 0, "unsure_ok": 0}

    records = []
    for row in rows:
        image = cv2.imread(os.path.join(dataset_dir, row["image_path"]))
        if image is None:
            continue
        truth = true_corners(row)
        entry = stats[row["bg_name"]]
        entry["images"] = entry["images"] + 1

        baseline = otsu_baseline_corners(image)
        if baseline is not None:
            entry["otsu_found"] = entry["otsu_found"] + 1
            baseline_mean, trash = corner_errors(baseline, truth)
            if baseline_mean <= SUCCESS_PIXELS:
                entry["otsu_mean_ok"] = entry["otsu_mean_ok"] + 1

        found, score = find_page_corners_with_score(image)
        if found is None:
            records.append((row["image_path"], None, None, score))
            continue
        entry["found"] = entry["found"] + 1
        mean_error, largest = corner_errors(found, truth)
        entry["errors"].append(mean_error)
        if mean_error <= SUCCESS_PIXELS:
            entry["mean_ok"] = entry["mean_ok"] + 1
        if largest <= SUCCESS_PIXELS:
            entry["max_ok"] = entry["max_ok"] + 1
        if score >= CONFIDENT:
            entry["confident"] = entry["confident"] + 1
            if mean_error <= SUCCESS_PIXELS:
                entry["confident_ok"] = entry["confident_ok"] + 1
        else:
            entry["unsure"] = entry["unsure"] + 1
            if mean_error <= SUCCESS_PIXELS:
                entry["unsure_ok"] = entry["unsure_ok"] + 1
        records.append((row["image_path"], mean_error, largest, score))

    print("")
    print("%-14s %7s %11s %11s %8s %11s %11s %12s" % ("background", "images", "otsu found", "otsu <=10px",
                                                     "found", "mean <=10px", "max <=10px", "median err"))
    totals = {"images": 0, "otsu_found": 0, "otsu_mean_ok": 0, "found": 0, "mean_ok": 0, "max_ok": 0,
              "confident": 0, "confident_ok": 0, "unsure": 0, "unsure_ok": 0}
    all_errors = []
    for name in backgrounds:
        entry = stats[name]
        median_error = median_of(entry["errors"])
        median_text = "none"
        if median_error is not None:
            median_text = "%.1f px" % median_error
        print("%-14s %7d %11d %11d %8d %11d %11d %12s" % (name, entry["images"], entry["otsu_found"],
                                                         entry["otsu_mean_ok"], entry["found"], entry["mean_ok"],
                                                         entry["max_ok"], median_text))
        for key in totals:
            totals[key] = totals[key] + entry[key]
        for error in entry["errors"]:
            all_errors.append(error)
    median_all = median_of(all_errors)
    median_all_text = "none"
    if median_all is not None:
        median_all_text = "%.1f px" % median_all
    print("%-14s %7d %11d %11d %8d %11d %11d %12s" % ("all", totals["images"], totals["otsu_found"],
                                                     totals["otsu_mean_ok"], totals["found"], totals["mean_ok"],
                                                     totals["max_ok"], median_all_text))
    print("")
    print("mean corner error <= %d px counts as success. 'max' is stricter: every corner within %d px." %
          (int(SUCCESS_PIXELS), int(SUCCESS_PIXELS)))
    print("score >= %.1f (confident): %d images, %d within 10 px (mean)" % (CONFIDENT, totals["confident"], totals["confident_ok"]))
    print("score <  %.1f (unsure):    %d images, %d within 10 px (mean)" % (CONFIDENT, totals["unsure"], totals["unsure_ok"]))

    # save every result so the numbers can be rechecked
    out_csv = open("smartdoc_results.csv", "w")
    out_csv.write("image_path,mean_error_px,max_error_px,score\n")
    for record in records:
        mean_text = ""
        max_text = ""
        if record[1] is not None:
            mean_text = "%.2f" % record[1]
            max_text = "%.2f" % record[2]
        out_csv.write("%s,%s,%s,%.3f\n" % (record[0], mean_text, max_text, record[3]))
    out_csv.close()

    # failure examples: confident but wrong first, then unsure and wrong
    confident_wrong = []
    unsure_wrong = []
    for record in records:
        if record[1] is None:
            continue
        if record[1] > SUCCESS_PIXELS:
            if record[3] >= CONFIDENT:
                confident_wrong.append(record)
            else:
                unsure_wrong.append(record)
    chosen = []
    step_c = max(1, len(confident_wrong) // 3)
    for i in range(0, len(confident_wrong), step_c):
        if len(chosen) < 3:
            chosen.append(confident_wrong[i])
    step_u = max(1, len(unsure_wrong) // 3)
    for i in range(0, len(unsure_wrong), step_u):
        if len(chosen) < 6:
            chosen.append(unsure_wrong[i])
    tiles = []
    for record in chosen:
        image = cv2.imread(os.path.join(dataset_dir, record[0]))
        row_match = None
        for row in rows:
            if row["image_path"] == record[0]:
                row_match = row
        truth = true_corners(row_match)
        found, trash = find_page_corners_with_score(image)
        draw_quad(image, truth, (0, 0, 255), 5)
        draw_quad(image, found, (0, 220, 0), 5)
        tile = cv2.resize(image, (640, 360))
        cv2.putText(tile, "score %.2f  err %.0f px" % (record[3], record[1]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(tile)
    if len(tiles) > 0:
        while len(tiles) < 6:
            tiles.append(np.zeros((360, 640, 3), dtype=np.uint8))
        cv2.imwrite("smartdoc_failures.png", np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]))
        print("saved smartdoc_failures.png and smartdoc_results.csv")
    else:
        print("saved smartdoc_results.csv (no failures to draw)")

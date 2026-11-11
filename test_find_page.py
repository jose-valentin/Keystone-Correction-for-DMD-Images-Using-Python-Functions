import math
import random

import cv2
import numpy as np

from correct_keystone import match_to_truth
from find_page import find_page_corners_with_score, otsu_baseline_corners

# Synthetic stand-in scenes for week 6. These are NOT real photos. Each scene is 1920 by 1080
# and tries to reproduce what broke Otsu in week 5: a lit table that is nearly as bright as the
# page, a bright object touching the page, a dark cable, and other clutter. The page is an A4
# shaped bright rectangle with text-like bars, seen in perspective, with blur and noise.

SCENE_WIDTH = 1920
SCENE_HEIGHT = 1080
SUCCESS_PIXELS = 10.0     # same limit as the week 5 test (within 10 px of the true corners)


def make_scene(seed):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # table color and a bright spot of light on it
    tones = [(150, 170, 200), (165, 175, 195), (140, 160, 175), (170, 178, 190), (120, 130, 150)]
    tone = tones[rng.randint(0, len(tones) - 1)]
    canvas = np.zeros((SCENE_HEIGHT, SCENE_WIDTH, 3), dtype=np.float32)
    for c in range(3):
        canvas[:, :, c] = tone[c]
    ys, xs = np.mgrid[0:SCENE_HEIGHT, 0:SCENE_WIDTH]
    light_x = rng.randint(500, 1400)
    light_y = rng.randint(250, 800)
    light_size = rng.randint(400, 700)
    gain = rng.uniform(0.10, 0.22)
    field = 0.92 + gain * np.exp(-(((xs - light_x) ** 2 + (ys - light_y) ** 2) / (2.0 * light_size ** 2)))
    field = field.astype(np.float32)

    # page corners in page space, then in the scene (rotation plus a random perspective jitter)
    page_h = SCENE_HEIGHT * rng.uniform(0.40, 0.62)
    page_w = page_h * 0.707
    center_x = rng.uniform(SCENE_WIDTH * 0.35, SCENE_WIDTH * 0.65)
    center_y = rng.uniform(SCENE_HEIGHT * 0.42, SCENE_HEIGHT * 0.58)
    angle = math.radians(rng.uniform(-25, 25))
    base = [(-page_w / 2, -page_h / 2), (page_w / 2, -page_h / 2), (page_w / 2, page_h / 2), (-page_w / 2, page_h / 2)]
    truth = []
    for corner in base:
        x = corner[0] * math.cos(angle) - corner[1] * math.sin(angle)
        y = corner[0] * math.sin(angle) + corner[1] * math.cos(angle)
        x = x + rng.uniform(-0.10, 0.10) * page_w
        y = y + rng.uniform(-0.08, 0.08) * page_h
        truth.append([center_x + x, center_y + y])
    truth = np.float32(truth)

    # clutter under the page: dark object, cable, bright mug touching the page
    mouse_x = rng.randint(100, 1800)
    mouse_y = rng.randint(100, 980)
    cv2.ellipse(canvas, (mouse_x, mouse_y), (rng.randint(50, 90), rng.randint(70, 120)), rng.randint(0, 90), 0, 360, (40, 40, 45), -1)
    cable_points = []
    cable_x = rng.randint(0, 600)
    cable_y = rng.randint(100, 980)
    for k in range(8):
        cable_points.append([cable_x + k * 250 + rng.randint(-60, 60), cable_y + rng.randint(-150, 150)])
    cv2.polylines(canvas, [np.int32(cable_points)], False, (50, 50, 55), rng.randint(5, 10))
    # mug: a bright disc that overlaps the page border from outside
    edge_index = rng.randint(0, 3)
    p = truth[edge_index]
    q = truth[(edge_index + 1) % 4]
    mid_x = float((p[0] + q[0]) / 2.0)
    mid_y = float((p[1] + q[1]) / 2.0)
    away_x = mid_x - center_x
    away_y = mid_y - center_y
    away_length = math.sqrt(away_x * away_x + away_y * away_y)
    radius = rng.randint(55, 85)
    mug_x = int(mid_x + away_x / away_length * radius * 0.85)
    mug_y = int(mid_y + away_y / away_length * radius * 0.85)
    mug_value = rng.randint(200, 232)
    cv2.circle(canvas, (mug_x, mug_y), radius, (mug_value, mug_value, mug_value), -1)
    cv2.circle(canvas, (mug_x, mug_y), int(radius * 0.7), (mug_value - 25, mug_value - 25, mug_value - 25), -1)
    for k in range(rng.randint(1, 3)):
        x0 = rng.randint(0, 1700)
        y0 = rng.randint(0, 900)
        value = rng.randint(60, 220)
        cv2.rectangle(canvas, (x0, y0), (x0 + rng.randint(60, 200), y0 + rng.randint(40, 150)), (value, value - 20, value + 10), -1)

    # the page: bright, with text-like bars, drawn at 1000 px high and warped into the scene
    src_h = 1000
    src_w = int(src_h * 0.707)
    page = np.zeros((src_h, src_w, 3), dtype=np.float32)
    page_value = rng.uniform(238, 252)
    page[:, :, :] = page_value
    page[:, :, 0] = page_value - rng.uniform(0, 6)
    y = 70
    while y < src_h - 70:
        bar_w = int(src_w * rng.uniform(0.35, 0.85))
        bar_value = rng.uniform(60, 120)
        cv2.rectangle(page, (60, y), (60 + bar_w, y + 9), (bar_value, bar_value, bar_value), -1)
        y = y + 26
    cv2.rectangle(page, (60, 60), (src_w - 60, 110), (90, 90, 90), -1)
    source_corners = np.float32([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]])
    transform = cv2.getPerspectiveTransform(source_corners, truth)
    warped = cv2.warpPerspective(page, transform, (SCENE_WIDTH, SCENE_HEIGHT))
    mask = cv2.warpPerspective(np.ones((src_h, src_w), dtype=np.float32), transform, (SCENE_WIDTH, SCENE_HEIGHT))
    for c in range(3):
        canvas[:, :, c] = canvas[:, :, c] * (1.0 - mask) + warped[:, :, c] * mask

    # lighting, blur, noise
    for c in range(3):
        canvas[:, :, c] = canvas[:, :, c] * field
    canvas = np.clip(canvas, 0, 255)
    sigma = rng.uniform(0.8, 2.0)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
    canvas = canvas + np_rng.normal(0, rng.uniform(2.0, 5.0), canvas.shape)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas, truth


def inside_scene(truth):
    for point in truth:
        if point[0] < 20 or point[0] > SCENE_WIDTH - 20 or point[1] < 20 or point[1] > SCENE_HEIGHT - 20:
            return False
    return True


def draw_quad(image, quad, color, thickness):
    points = np.int32(quad).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, thickness)


if __name__ == "__main__":
    scene_count = 200
    baseline_found = 0
    baseline_within = 0
    baseline_errors = []
    new_found = 0
    new_within = 0
    new_errors = []
    examples = []

    made = 0
    seed = 1000
    while made < scene_count:
        seed = seed + 1
        image, truth = make_scene(seed)
        if not inside_scene(truth):
            continue
        made = made + 1

        baseline = otsu_baseline_corners(image)
        baseline_error = None
        if baseline is not None:
            baseline_found = baseline_found + 1
            baseline_error, trash = match_to_truth(baseline, truth)
            baseline_errors.append(baseline_error)
            if baseline_error <= SUCCESS_PIXELS:
                baseline_within = baseline_within + 1

        found, score = find_page_corners_with_score(image)
        new_error = None
        if found is not None:
            new_found = new_found + 1
            new_error, trash = match_to_truth(found, truth)
            new_errors.append(new_error)
            if new_error <= SUCCESS_PIXELS:
                new_within = new_within + 1

        if len(examples) < 6:
            tile = image.copy()
            draw_quad(tile, truth, (0, 0, 255), 4)
            if baseline is not None:
                draw_quad(tile, baseline, (255, 120, 0), 4)
            if found is not None:
                draw_quad(tile, found, (0, 220, 0), 4)
            examples.append(cv2.resize(tile, (640, 360)))

    baseline_errors.sort()
    new_errors.sort()
    print("synthetic scenes:", scene_count, "(1920 by 1080, NOT real photos)")
    print("success = four corners found and mean corner error <= %d px" % int(SUCCESS_PIXELS))
    print("")
    print("Otsu baseline:   found", baseline_found, " within 10 px", baseline_within,
          " median error among found", round(baseline_errors[len(baseline_errors) // 2], 1) if len(baseline_errors) > 0 else None, "px")
    print("Rectangle score: found", new_found, " within 10 px", new_within,
          " median error among found", round(new_errors[len(new_errors) // 2], 1) if len(new_errors) > 0 else None, "px")

    top = np.hstack(examples[0:3])
    bottom = np.hstack(examples[3:6])
    cv2.imwrite("scene_examples.png", np.vstack([top, bottom]))

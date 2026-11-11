# Keystone correction for DMD projector images

Python and OpenCV code for a university research project (UPRM, undergraduate research course).
The goal is to find a projected image (or a page) in a camera photo, find its four corners, and
straighten it with a perspective transform. This is the keystone distortion problem for DMD
(Digital Micromirror Device) projectors.

This is a work in progress. Right now the code finds the four corners and straightens the
shape. Finding the projector angles (yaw, pitch) from the corners is the next step.

## What is in here

| File | What it does |
| --- | --- |
| `find_page_simple.py` | The corner finder. Tries many brightness cutoffs on each color channel, keeps every 4 sided shape, and picks the one closest to square. |
| `correct_keystone.py` | Corner ordering, perspective transform, and helpers for matching corners to the true ones. |
| `demo_simple.py` | One photo in, a picture out with the found corners and the straightened result. |
| `sim.py` | Simulated projector test. Makes a distorted "photo" with known corners and checks the finder. |
| `find_page.py` | An earlier, more complex finder (kept for comparison). |
| `test_find_page.py` | Tests on 200 synthetic scenes. |
| `evaluate_smartdoc.py` | Tests on real photos from the SmartDoc dataset. |
| `make_figures.py` | Helpers that draw the steps of the finder. |

## How to run

```
pip install -r requirements.txt
python3 sim.py
python3 test_find_page.py
python3 demo_simple.py <photo> out.png
```

`sim.py` needs no photos. On the simulated projector frame the found corners are within about
1.5 pixels of the true ones.

## Results so far

- Synthetic scenes (200, 1920 by 1080, not real photos): the rectangle score finder found the
  corners within 10 px in 188 of 200. A plain Otsu threshold baseline got 11 of 200.
- Real photos (SmartDoc 2015 Challenge 1, a subset of 752 photos): `find_page_simple.py` found
  the corners within 10 px of the true ones in 505 of 752 (67.2%).

The finder is not reliable on every photo yet.

## Datasets

The photos are not in this repo. SmartDoc 2015 Challenge 1 (Burie, Chazalon, Coustaty,
Eskenazi, Luqman, Mehri, Nayef, Ogier, Prum, Rusinol) is licensed CC BY 4.0. Download it from
the ICDAR 2015 SmartDoc challenge page and point `evaluate_smartdoc.py` at the folder.

## Sources

Where an idea comes from a public source it is named at the top of the file, and each file says
whether the code was ported directly or adapted with changes.

- OpenCV `squares.py` sample: https://github.com/opencv/opencv/blob/4.x/samples/python/squares.py
- PyImageSearch four point transform: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

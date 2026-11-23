import base64
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog

import analysis
from make_calibration_image import make_calibration_image

# Keystone dashboard. Open a photo, see the found outline, the straightened result, the
# corner numbers and a confidence flag. Plain Tkinter, flat square boxes, no extra libraries.
#   python3 dashboard.py [photo]
# All the image work is in analysis.py. This file is only the window.

PANEL_WIDTH = 520
PANEL_HEIGHT = 390
BACKGROUND = "#ffffff"
PANEL_COLOR = "#f0f0f0"
BORDER_COLOR = "#000000"
FONT = ("Arial", 10)
FONT_BOLD = ("Arial", 10, "bold")
FONT_MONO = ("Consolas", 10)


class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keystone Dashboard")
        self.root.configure(bg=BACKGROUND)
        self.root.resizable(False, False)

        self.photo = None
        self.photo_path = ""
        self.result = None
        self.worker_result = None
        self.worker_done = False
        self.image_refs = []      # keeps the Tk pictures alive

        self.build_buttons()
        self.build_panels()
        self.build_results_box()
        self.set_lines(["Open a photo to begin."])

        self.root.bind("<Control-o>", self.on_open)

    # ---- layout ----
    def make_button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, font=FONT, bg=BACKGROUND,
                         activebackground="#e0e0e0", relief="solid", bd=1, padx=12, pady=4,
                         cursor="hand2")

    def build_buttons(self):
        bar = tk.Frame(self.root, bg=BACKGROUND)
        bar.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6))
        self.make_button(bar, "Open photo...", self.on_open).pack(side="left", padx=(0, 8))
        self.make_button(bar, "Save straightened...", self.on_save_straightened).pack(side="left", padx=(0, 8))
        self.make_button(bar, "Save outline picture...", self.on_save_outline).pack(side="left", padx=(0, 8))
        self.make_button(bar, "Save calibration frame...", self.on_save_calibration).pack(side="left")

    def make_panel(self, column, title):
        frame = tk.Frame(self.root, bg=BACKGROUND)
        frame.grid(row=1, column=column, padx=12, pady=6, sticky="n")
        tk.Label(frame, text=title, font=FONT_BOLD, bg=BACKGROUND, anchor="w").pack(fill="x")
        box = tk.Frame(frame, bg=PANEL_COLOR, highlightbackground=BORDER_COLOR,
                       highlightthickness=1, width=PANEL_WIDTH, height=PANEL_HEIGHT)
        box.pack()
        box.pack_propagate(False)
        label = tk.Label(box, bg=PANEL_COLOR, font=FONT, text="")
        label.pack(expand=True)
        return label

    def build_panels(self):
        self.original_label = self.make_panel(0, "Photo with found outline")
        self.straight_label = self.make_panel(1, "Straightened")

    def build_results_box(self):
        frame = tk.Frame(self.root, bg=BACKGROUND)
        frame.grid(row=2, column=0, columnspan=2, padx=12, pady=(6, 12), sticky="w")
        tk.Label(frame, text="Results", font=FONT_BOLD, bg=BACKGROUND, anchor="w").pack(fill="x")
        self.results_text = tk.Text(frame, width=132, height=13, font=FONT_MONO, bg=PANEL_COLOR,
                                    relief="flat", highlightbackground=BORDER_COLOR,
                                    highlightthickness=1, state="disabled", wrap="word")
        self.results_text.pack()

    # ---- showing things ----
    def set_lines(self, lines):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")

    def show_image(self, label, image, empty_text):
        if image is None:
            label.configure(image="", text=empty_text)
            return
        png = analysis.to_png_bytes(image, PANEL_WIDTH - 4, PANEL_HEIGHT - 4)
        if png is None:
            label.configure(image="", text=empty_text)
            return
        picture = tk.PhotoImage(data=base64.b64encode(png))
        self.image_refs.append(picture)
        if len(self.image_refs) > 6:
            self.image_refs.pop(0)
        label.configure(image=picture, text="")

    # ---- actions ----
    def on_open(self, event=None):
        path = filedialog.askopenfilename(
            title="Open photo",
            filetypes=[("Pictures", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self.open_path(path)

    def open_path(self, path):
        photo = analysis.load_photo(path)
        if photo is None:
            self.set_lines(["Could not read this file as a picture:", path])
            return
        self.photo = photo
        self.photo_path = path
        self.result = None
        self.show_image(self.original_label, photo, "")
        self.show_image(self.straight_label, None, "Working...")
        self.set_lines(["Working on " + os.path.basename(path) + " ..."])
        self.start_worker(photo)

    # The analysis can take a few seconds on a big photo, so it runs in a separate thread
    # and the window checks every 100 ms whether it is done.
    def start_worker(self, photo):
        self.worker_done = False
        self.worker_result = None
        thread = threading.Thread(target=self.run_worker, args=(photo,))
        thread.daemon = True
        thread.start()
        self.root.after(100, self.check_worker)

    def run_worker(self, photo):
        self.worker_result = analysis.analyze_photo(photo)
        self.worker_done = True

    def check_worker(self):
        if not self.worker_done:
            self.root.after(100, self.check_worker)
            return
        self.result = self.worker_result
        self.show_image(self.original_label, self.result.outline_image, "")
        self.show_image(self.straight_label, self.result.straightened, "Nothing to show")
        self.set_lines(analysis.result_lines(self.result, os.path.basename(self.photo_path)))

    def save_with_dialog(self, image, title, default_name):
        if image is None:
            self.set_lines(["Nothing to save yet."])
            return
        path = filedialog.asksaveasfilename(title=title, defaultextension=".png", initialfile=default_name,
                                            filetypes=[("PNG picture", "*.png"), ("JPEG picture", "*.jpg")])
        if not path:
            return
        if analysis.save_image(path, image):
            self.set_lines(["Saved " + path])
        else:
            self.set_lines(["Could not save " + path])

    def on_save_straightened(self):
        image = None
        if self.result is not None:
            image = self.result.straightened
        self.save_with_dialog(image, "Save straightened picture", "straightened.png")

    def on_save_outline(self):
        image = None
        if self.result is not None:
            image = self.result.outline_image
        self.save_with_dialog(image, "Save outline picture", "outline.png")

    def on_save_calibration(self):
        self.save_with_dialog(make_calibration_image(), "Save calibration frame", "calibration_frame.png")


def main():
    root = tk.Tk()
    app = DashboardApp(root)
    if len(sys.argv) > 1:
        app.open_path(sys.argv[1])
    root.mainloop()


if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import csv
import json
import os
import sys

# Rate a folder of face pictures from 1 to 10 into ratings.csv.
# Run: python dataset_rater.py [folder]
# Keys: 1-9 and 0 (= 10) rate right away, Enter submits the slider,
#       Backspace or Left = undo, S or Right = skip.

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Keep the files next to this script, no matter which folder you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RATINGS_FILE = os.path.join(SCRIPT_DIR, "ratings.csv")
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "dataset_rater_settings.json")

HEADER = ["filename", "rating"]

# Pictures are shrunk or enlarged to fit inside a square of this many pixels, keeping their shape
BOX_SIZE = 400


# ---------- Plain functions (no window), so they can be tested ----------

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def load_ratings(path):
    # Returns a list of [filename, rating] rows. Works with and without the header row,
    # because ratings.csv files made by the old tools have no header
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if len(row) >= 2]
    if rows and [cell.strip().lower() for cell in rows[0][:2]] == HEADER:
        rows = rows[1:]
    return [[row[0], row[1]] for row in rows]


def save_ratings(path, rows):
    # Writes the whole file every time. It's only a few hundred lines, and it means an old
    # headerless file gets a header, and undo can remove the last line.
    # Write to a temporary file first, so a crash halfway can't leave a half-written ratings.csv
    temp = path + ".tmp"
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    os.replace(temp, path)


def next_unrated(images, rows, start=0):
    # Index of the first picture from start on that isn't in the ratings yet, or len(images) if there is none.
    # Looking at the names instead of the last line means skipped pictures and undo don't confuse it
    rated = {row[0] for row in rows}
    for i in range(start, len(images)):
        if images[i] not in rated:
            return i
    return len(images)


def undo_last(images, rows):
    # Removes the last row and returns the index of its picture, so you can rate it again.
    # Returns None (and removes nothing) if there is no row, or the picture isn't in this folder
    if not rows or rows[-1][0] not in images:
        return None
    filename = rows.pop()[0]
    return images.index(filename)


def count_rated(images, rows):
    rated = {row[0] for row in rows}
    return sum(1 for name in images if name in rated)


def fit_size(width, height, box):
    # The biggest size that fits inside a box x box square without stretching the picture
    scale = min(box / width, box / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_last_folder():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f).get("last_folder")
    except (OSError, ValueError, AttributeError):
        # No settings file yet, or it's broken. Then we just ask for a folder
        return None


def save_last_folder(folder):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_folder": folder}, f)
    except OSError:
        pass  # Not being able to remember the folder isn't worth stopping for


def choose_folder():
    # The folder on the command line, else the one used last time, else ask
    if len(sys.argv) > 1:
        return sys.argv[1]
    last = load_last_folder()
    if last and os.path.isdir(last):
        return last
    return filedialog.askdirectory(title="Pick the folder with the face pictures")


# ---------- The window ----------

class DatasetRater:
    def __init__(self, root, folder):
        self.root = root
        self.root.title("Dataset Rater")
        self.folder = folder
        self.images = list_images(folder)
        self.rows = load_ratings(RATINGS_FILE)
        self.index = next_unrated(self.images, self.rows)

        self.progress_label = ttk.Label(root)
        self.progress_label.pack(pady=(8, 0))

        self.canvas = tk.Canvas(root, width=BOX_SIZE, height=BOX_SIZE, highlightthickness=0)
        self.canvas.pack(padx=10, pady=8)

        self.name_label = ttk.Label(root)
        self.name_label.pack()

        # Starts at 1, otherwise the scale starts at 0, which is below the 1-10 range.
        # takefocus=False keeps the keyboard on the window, so Left/Right undo and skip instead of moving the slider
        self.scale = ttk.Scale(root, from_=1, to=10, orient="horizontal", length=250, value=5,
                               command=self.show_slider_value, takefocus=False)
        self.scale.pack(pady=(8, 0))

        # Show the number the scale is on, ttk.Scale doesn't do that by itself
        self.slider_label = ttk.Label(root, text="5")
        self.slider_label.pack()

        ttk.Button(root, text="Submit", command=self.submit_slider, takefocus=False).pack(pady=4)
        ttk.Label(root, text="1-9, 0 = 10: rate   Enter: submit slider   Backspace/Left: undo   S/Right: skip").pack(padx=10, pady=(0, 8))

        for i in range(1, 10):
            root.bind(str(i), lambda event, i=i: self.rate(i))
        root.bind("0", lambda event: self.rate(10))
        root.bind("<Return>", lambda event: self.submit_slider())
        root.bind("<BackSpace>", lambda event: self.undo())
        root.bind("<Left>", lambda event: self.undo())
        root.bind("s", lambda event: self.skip())
        root.bind("S", lambda event: self.skip())
        root.bind("<Right>", lambda event: self.skip())

        self.show_picture()

    def show_slider_value(self, value):
        self.slider_label.config(text=str(round(float(value))))

    def show_picture(self):
        self.progress_label.config(text=f"{count_rated(self.images, self.rows)} / {len(self.images)}")
        self.canvas.delete("all")

        if self.index >= len(self.images):
            # Don't close the window, so the last rating can still be undone
            self.photo = None
            self.name_label.config(text="")
            self.canvas.create_text(BOX_SIZE / 2, BOX_SIZE / 2, width=BOX_SIZE - 20, justify="center",
                                    text="End of the folder. Skipped pictures come back next time.\n"
                                         "Backspace = undo, or close the window.")
            return

        filename = self.images[self.index]
        self.name_label.config(text=filename)
        try:
            with Image.open(os.path.join(self.folder, filename)) as image:
                image = image.convert("RGB")
                image = image.resize(fit_size(image.width, image.height, BOX_SIZE), Image.Resampling.LANCZOS)
        except OSError:
            self.canvas.create_text(BOX_SIZE / 2, BOX_SIZE / 2, text="Could not open this picture.\nPress S to skip it.")
            return

        # Keep a reference, otherwise tkinter throws the picture away and shows nothing
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(BOX_SIZE / 2, BOX_SIZE / 2, anchor="center", image=self.photo)

    def rate(self, rating):
        if self.index >= len(self.images):
            return
        self.rows.append([self.images[self.index], rating])
        save_ratings(RATINGS_FILE, self.rows)
        self.index = next_unrated(self.images, self.rows, self.index + 1)
        self.show_picture()

    def submit_slider(self):
        self.rate(round(self.scale.get()))

    def skip(self):
        if self.index >= len(self.images):
            return
        self.index = next_unrated(self.images, self.rows, self.index + 1)
        self.show_picture()

    def undo(self):
        index = undo_last(self.images, self.rows)
        if index is None:
            self.root.bell()
            return
        save_ratings(RATINGS_FILE, self.rows)
        self.index = index
        self.show_picture()


def main():
    root = tk.Tk()
    root.withdraw()  # Hidden until we know there is something to show

    folder = choose_folder()
    if not folder:
        root.destroy()
        return
    if not list_images(folder):
        messagebox.showerror("Dataset Rater", f"No .jpg or .png pictures found in\n{folder}")
        root.destroy()
        return
    save_last_folder(os.path.abspath(folder))

    DatasetRater(root, folder)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()

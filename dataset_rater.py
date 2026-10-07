import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import csv
import json
import os
import sys
import time

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

# Excel or OneDrive can lock ratings.csv for a moment, so saving tries this many times, this many seconds apart
SAVE_TRIES = 5
SAVE_WAIT = 0.1


class RatingsFileError(Exception):
    # ratings.csv has a line we don't understand. We stop instead of guessing, so no rating is ever lost
    pass


# ---------- Plain functions (no window), so they can be tested ----------

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def read_text(path):
    # Files from this tool are utf-8, but the old tools wrote Windows' own encoding (cp1252),
    # so a name like bilde_æøå.jpg isn't valid utf-8 there. utf-8-sig also removes the BOM Excel adds
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        raise RatingsFileError("ratings.csv isn't a text file (not utf-8 or cp1252).")


def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_ratings(text):
    # Returns (rows, delimiter). rows is a list of [filename, rating].
    # Excel with Norwegian settings saves with ; instead of , so look at the first line to see which one is used
    lines = text.splitlines()
    firstLine = next((line for line in lines if line.strip()), "")
    delimiter = ";" if ";" in firstLine else ","

    rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cells = next(csv.reader([line], delimiter=delimiter))
        except csv.Error:
            cells = []

        # The old tools didn't quote names, so "smith, john.jpg,8" became three cells. The rating is always last
        if len(cells) >= 2:
            filename, rating = delimiter.join(cells[:-1]), cells[-1]
            if not rows and [filename.strip().lower(), rating.strip().lower()] == HEADER:
                continue  # The header row (files from the old tools don't have one)
            if is_number(rating):
                rows.append([filename, rating.strip()])
                continue

        raise RatingsFileError(f"Line {number} of ratings.csv isn't 'filename{delimiter}rating':\n{line}")
    return rows, delimiter


def load_ratings(path):
    # Returns (rows, delimiter). A missing file is the same as no ratings yet
    if not os.path.isfile(path):
        return [], ","
    return parse_ratings(read_text(path))


def save_ratings(path, rows, delimiter=","):
    # Writes the whole file every time. It's only a few hundred lines, and it means an old
    # headerless file gets a header, and undo can remove the last line.
    # Write to a temporary file first, so a crash halfway can't leave a half-written ratings.csv.
    # Raises OSError if the file stays locked, and then ratings.csv is left as it was
    temp = path + ".tmp"
    try:
        with open(temp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(HEADER)
            writer.writerows(rows)
        for attempt in range(SAVE_TRIES):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == SAVE_TRIES - 1:
                    raise
                time.sleep(SAVE_WAIT)
    except OSError:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def next_unrated(images, rows, start=0):
    # Index of the first picture from start on that isn't in the ratings yet, or len(images) if there is none.
    # Looking at the names instead of the last line means skipped pictures and undo don't confuse it
    rated = {row[0] for row in rows}
    for i in range(start, len(images)):
        if images[i] not in rated:
            return i
    return len(images)


def undo_last(images, rows):
    # Returns (rows without the last one, index of its picture), so you can rate it again.
    # Returns (rows, None) if there is no row, or the picture isn't in this folder. rows itself isn't changed
    if not rows or rows[-1][0] not in images:
        return rows, None
    return rows[:-1], images.index(rows[-1][0])


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
            settings = json.load(f)
    except (OSError, ValueError):
        # No settings file yet, or it's broken. Then we just ask for a folder
        return None
    folder = settings.get("last_folder") if isinstance(settings, dict) else None
    # Someone may have edited the file by hand, so only accept text
    return folder if isinstance(folder, str) else None


def save_last_folder(folder):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_folder": folder}, f)
    except OSError:
        pass  # Not being able to remember the folder isn't worth stopping for


def choose_folder():
    # The folder on the command line, else the one used last time. If that's missing or has
    # no pictures, ask until we get a folder with pictures. Returns None if you press Cancel
    folder = sys.argv[1] if len(sys.argv) > 1 else load_last_folder()
    title = "Pick the folder with the face pictures"
    while not folder or not list_images(folder):
        if folder:
            title = f"No .jpg or .png pictures in {folder}. Pick another folder"
        folder = filedialog.askdirectory(title=title)
        if not folder:
            return None
    return folder


# ---------- The window ----------

class DatasetRater:
    def __init__(self, root, folder, rows, delimiter):
        self.root = root
        self.root.title("Dataset Rater")
        self.folder = folder
        self.images = list_images(folder)
        self.rows = rows
        self.delimiter = delimiter
        self.index = next_unrated(self.images, self.rows)
        self.broken = False  # True while the picture on screen couldn't be opened

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
        self.broken = False

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
            # You can't rate what you can't see, so only skipping works on this one
            self.broken = True
            self.canvas.create_text(BOX_SIZE / 2, BOX_SIZE / 2, text="Could not open this picture.\nPress S to skip it.")
            return

        # Keep a reference, otherwise tkinter throws the picture away and shows nothing
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(BOX_SIZE / 2, BOX_SIZE / 2, anchor="center", image=self.photo)

    def save(self, rows):
        # Only change self.rows after the file is saved, so the window and ratings.csv always agree
        try:
            save_ratings(RATINGS_FILE, rows, self.delimiter)
        except OSError:
            messagebox.showerror("Dataset Rater", "ratings.csv is locked, close Excel and try again.")
            return False
        self.rows = rows
        return True

    def rate(self, rating):
        if self.index >= len(self.images) or self.broken:
            return
        if self.save(self.rows + [[self.images[self.index], rating]]):
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
        rows, index = undo_last(self.images, self.rows)
        if index is None:
            self.root.bell()
            return
        if self.save(rows):
            self.index = index
            self.show_picture()


def main():
    root = tk.Tk()
    root.withdraw()  # Hidden until we know there is something to show

    try:
        rows, delimiter = load_ratings(RATINGS_FILE)
    except RatingsFileError as e:
        # Saving would mean leaving out the line we don't understand, so don't start at all
        messagebox.showerror("Dataset Rater", f"{e}\n\nFix or remove that line in\n{RATINGS_FILE}\nand start again.")
        root.destroy()
        return

    folder = choose_folder()
    if not folder:
        root.destroy()
        return
    save_last_folder(os.path.abspath(folder))

    DatasetRater(root, folder, rows, delimiter)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()

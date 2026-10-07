import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

# Folder with the face pictures to rate. If it's missing or empty you get asked to pick a folder.
IMAGE_FOLDER = "C:/Users/Damlu/OneDrive/Pictures/faces/01000-20230822T135533Z-001/01000"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Keep ratings.csv next to this script, no matter which folder you run it from
RATINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings.csv")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))

class FaceRatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Rating App")

        # Define the path to the folder containing the images
        self.image_path = IMAGE_FOLDER

        # Get a list of image filenames in the folder
        self.image_list = list_images(self.image_path)
        if not self.image_list:
            self.image_path = filedialog.askdirectory(title="Pick the folder with the face pictures")
            self.image_list = list_images(self.image_path) if self.image_path else []
        if not self.image_list:
            messagebox.showerror("Face Rating App", "No .jpg or .png pictures found.")
            self.root.destroy()
            return

        self.current_index = self.load_last_rated_index()
        if self.current_index >= len(self.image_list):
            messagebox.showinfo("Face Rating App", "All pictures in this folder are already rated.")
            self.root.destroy()
            return

        # Create and pack GUI elements
        self.label = ttk.Label(self.root, text="Rate the beauty of this face:")
        self.label.pack()

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        # Starts at 1, otherwise the scale starts at 0, which is below the 1-10 range
        self.rating_scale = ttk.Scale(self.root, from_=1, to=10, orient="horizontal", value=1,
                                      command=self.show_rating)
        self.rating_scale.pack()

        # Show the number the scale is on, ttk.Scale doesn't do that by itself
        self.rating_label = ttk.Label(self.root, text="1")
        self.rating_label.pack()

        self.submit_button = ttk.Button(self.root, text="Submit Rating", command=self.submit_rating)
        self.submit_button.pack()

        # Load the first image
        self.load_image()

    def load_last_rated_index(self):
        # return 0
        try:
            with open(RATINGS_FILE, "r") as f:
                lines = [line for line in f if line.strip()]
                if lines:
                    last_line = lines[-1].strip()
                    last_image_filename = last_line.split(",")[0]
                    last_index = self.image_list.index(last_image_filename)
                    return last_index + 1
                else:
                    return 0  # Default to 0 if the file is empty
        except FileNotFoundError:
            return 0  # Default to 0 if the file is missing
        except ValueError:
            return 0  # The last rated picture isn't in this folder, start from the beginning

    def load_image(self):
        # Get the filename of the current image
        image_filename = self.image_list[self.current_index]

        # Open and resize the image using PIL
        with Image.open(os.path.join(self.image_path, image_filename)) as image:
            image = image.convert("RGB").resize((300, 300), Image.Resampling.LANCZOS)

        # Convert the image to PhotoImage format for Tkinter
        self.photo = ImageTk.PhotoImage(image)

        # Create an image canvas and display the image
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def show_rating(self, value):
        self.rating_label.config(text=str(round(float(value))))

    def submit_rating(self):
        # Get the filename of the current image
        image_filename = self.image_list[self.current_index]

        # Get the selected beauty rating from the scale
        beauty_rating = self.rating_scale.get()

        rounded_rating = round(beauty_rating)

        # Open the CSV file in append mode and write the rating information
        with open(RATINGS_FILE, "a") as f:
            f.write(f"{image_filename},{rounded_rating}\n")

        # Move to the next image and reset the rating scale
        self.current_index += 1
        if self.current_index < len(self.image_list):
            self.load_image()
            self.rating_scale.set(1)  # Reset rating scale
            self.show_rating(1)
        else:
            # Close the application when all images are rated
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRatingApp(root)
    root.mainloop()
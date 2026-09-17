import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class FaceRatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Rating App")

        # Define the path to the folder containing the images
        self.image_path = "C:/Users/Damlu/OneDrive/Pictures/faces/01000-20230822T135533Z-001/01000"

        # Get a list of image filenames in the folder
        self.image_list = sorted([f for f in os.listdir(self.image_path) if f.endswith(".jpg") or f.endswith(".png")])

        self.current_index = self.load_last_rated_index()

        # Create and pack GUI elements
        self.label = ttk.Label(self.root, text="Rate the beauty of this face:")
        self.label.pack()

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        self.rating_scale = ttk.Scale(self.root, from_=1, to=10, orient="horizontal")
        self.rating_scale.pack()

        self.submit_button = ttk.Button(self.root, text="Submit Rating", command=self.submit_rating)
        self.submit_button.pack()

        # Load the first image
        self.load_image()

    def load_last_rated_index(self):
        # return 0
        try:
            with open("ratings.csv", "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    last_image_filename = last_line.split(",")[0]
                    last_index = self.image_list.index(last_image_filename)
                    return last_index + 1
                else:
                    return 0  # Default to 0 if the file is empty
        except FileNotFoundError:
            return 0  # Default to 0 if the file is missing

    def load_image(self):
        # Get the filename of the current image
        image_filename = self.image_list[self.current_index]

        # Open and resize the image using PIL
        image = Image.open(os.path.join(self.image_path, image_filename))
        image = image.resize((300, 300), Image.LANCZOS)

        # Convert the image to PhotoImage format for Tkinter
        self.photo = ImageTk.PhotoImage(image)

        # Create an image canvas and display the image
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def submit_rating(self):
        # Get the filename of the current image
        image_filename = self.image_list[self.current_index]

        # Get the selected beauty rating from the scale
        beauty_rating = self.rating_scale.get()

        rounded_rating = round(beauty_rating)

        # Open the CSV file in append mode and write the rating information
        with open("ratings.csv", "a") as f:
            f.write(f"{image_filename},{rounded_rating}\n")

        # Move to the next image and reset the rating scale
        self.current_index += 1
        if self.current_index < len(self.image_list):
            self.load_image()
            self.rating_scale.set(1)  # Reset rating scale
        else:
            # Close the application when all images are rated
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRatingApp(root)
    root.mainloop()
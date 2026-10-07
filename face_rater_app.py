import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import cv2
    from PIL import Image, ImageTk
    import landmarkdetect as ld
except ImportError as e:
    # Started by double-clicking there's no terminal to show the error in, so show it in a window
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("AI Face Rater", f"Missing Python package: {e.name}\n\n"
                                          "Install the packages with:\npip install -r requirements.txt")
    sys.exit(1)


# Which camera to use, 0 is the PC's default camera
CAMERA_INDEX = 0
# The live preview looks for the face in a copy this wide, which is much faster than the full picture
PREVIEW_DETECT_WIDTH = 320
# How often the live preview updates, in milliseconds
PREVIEW_DELAY = 30

GOOD = "#1a7f37"
BAD = "#c62828"
NEUTRAL = "#555555"


class Camera:
    # Reads frames from the camera in the background, so the window doesn't freeze while waiting for the next one

    def __init__(self, index):
        self.index = index
        self.frame = None
        self.opened = None  # None while starting, then True or False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        # DirectShow opens much faster than the default on Windows
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.opened = False
            return
        self.opened = True

        fails = 0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    self.opened = False
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self.lock:
                self.frame = frame
        cap.release()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


class FaceRaterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Rater")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.camera = None
        self.mode = "camera"  # "camera" shows the live preview, "result" shows a rated picture
        self.currentPicture = None
        self.animation = None

        # Picture area fits the screen (leaving room for the buttons and score below it), 4:3 like most webcams
        screenH = self.root.winfo_screenheight()
        self.viewH = max(300, min(720, int(screenH * 0.55), screenH - 480))
        self.viewW = self.viewH * 4 // 3

        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Score.TLabel", font=("Segoe UI", 26, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 11), padding=(14, 6))

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="AI Face Rater", style="Title.TLabel").pack(anchor="w")

        viewFrame = tk.Frame(main, width=self.viewW, height=self.viewH, bg="#202020")
        viewFrame.pack(pady=(8, 6))
        viewFrame.pack_propagate(False)
        self.view = tk.Label(viewFrame, bg="#202020", fg="white", font=("Segoe UI", 12))
        self.view.pack(fill="both", expand=True)

        self.status = tk.Label(main, text="", font=("Segoe UI", 13, "bold"), fg=NEUTRAL, wraplength=self.viewW)
        self.status.pack(pady=(0, 6))

        genderRow = ttk.Frame(main)
        genderRow.pack(pady=(0, 6))
        ttk.Label(genderRow, text="Compare with the model face of a:").pack(side="left", padx=(0, 8))
        self.gender = tk.StringVar(value="boy")
        for text, value in (("Boy", "boy"), ("Girl", "girl")):
            ttk.Radiobutton(genderRow, text=text, value=value, variable=self.gender,
                            command=self.gender_changed).pack(side="left", padx=4)

        buttons = ttk.Frame(main)
        buttons.pack(pady=(0, 6))
        self.takeButton = ttk.Button(buttons, text="Take photo  (Space)", style="Big.TButton",
                                     command=self.take_photo, state="disabled")
        self.takeButton.pack(side="left", padx=4)
        self.pickButton = ttk.Button(buttons, text="Pick a photo...", style="Big.TButton",
                                     command=self.pick_photo, state="disabled")
        self.pickButton.pack(side="left", padx=4)
        self.backButton = ttk.Button(buttons, text="Back to camera  (Esc)", style="Big.TButton",
                                     command=self.back_to_camera)

        # Empty score and details still take up their space, so the window doesn't change size when a result shows up
        self.scoreLabel = ttk.Label(main, text=" ", style="Score.TLabel")
        self.scoreLabel.pack()
        self.detailLabel = ttk.Label(main, text="\n", foreground=NEUTRAL, justify="center")
        self.detailLabel.pack()

        self.root.bind("<space>", lambda e: self.take_photo())
        self.root.bind("<Escape>", lambda e: self.back_to_camera())
        self.root.bind("<Control-o>", lambda e: self.pick_photo())

        self.view.config(text="Loading the face model...")
        self.root.after(50, self.start)

    # ---------- Starting and closing ----------

    def start(self):
        try:
            ld.load_predictor()
        except FileNotFoundError as e:
            messagebox.showerror("AI Face Rater", str(e))
            self.root.destroy()
            return
        self.pickButton.config(state="normal")
        self.view.config(text="Starting the camera...")
        self.camera = Camera(CAMERA_INDEX)
        self.update_preview()

    def close(self):
        if self.camera is not None:
            self.camera.stop()
        self.root.destroy()

    # ---------- Live camera preview ----------

    def update_preview(self):
        if self.mode == "camera":
            if self.camera.opened is False:
                self.view.config(image="", text="No camera found.\nUse \"Pick a photo...\" instead.")
                self.photo = None
                self.takeButton.config(state="disabled")
                self.set_status("", NEUTRAL)
            else:
                frame = self.camera.latest()
                if frame is not None:
                    self.show_preview_frame(frame)
        self.root.after(PREVIEW_DELAY, self.update_preview)

    def show_preview_frame(self, frame):
        # Mirror it, so moving your head to the left moves it to the left on the screen, like a mirror
        frame = cv2.flip(frame, 1)
        self.takeButton.config(state="normal")

        h, w = frame.shape[:2]
        found = ld.landmark_detect(frame, detectScale=min(1.0, PREVIEW_DETECT_WIDTH / w))
        if found is None:
            self.set_status("No face found. Face the camera!", BAD)
            banner = "FACE THE CAMERA"
        else:
            xList, yList = found
            problem = ld.facing_problem(xList, yList, frame.shape)
            if problem is None:
                self.set_status("Looking good! Press \"Take photo\" (or Space).", GOOD)
                banner = None
                color = (80, 200, 80)
            else:
                self.set_status(f"Face the camera! You are facing {problem}.", BAD)
                banner = "FACE THE CAMERA"
                color = (60, 60, 230)
            for x, y in zip(xList, yList):
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)

        if banner is not None:
            # A see-through red bar at the top of the picture, so you see it while looking at the camera
            barH = max(30, h // 9)
            bar = frame[:barH].copy()
            bar[:] = (40, 40, 200)
            frame[:barH] = cv2.addWeighted(frame[:barH], 0.35, bar, 0.65, 0)
            scale = barH / 45
            (textW, textH), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(frame, banner, ((w - textW) // 2, (barH + textH) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 2, cv2.LINE_AA)

        self.show_picture(frame)

    # ---------- Rating ----------

    def take_photo(self):
        if self.mode != "camera" or self.camera is None or not self.camera.opened:
            return
        frame = self.camera.latest()
        if frame is None:
            return
        # Rate the same mirrored picture you saw in the preview
        self.rate(cv2.flip(frame, 1))

    def pick_photo(self):
        if str(self.pickButton["state"]) == "disabled":
            return
        path = filedialog.askopenfilename(
            title="Pick a photo of a face",
            filetypes=[("Pictures", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            picture = ld.read_image(path)
        except ld.FaceError as e:
            messagebox.showerror("AI Face Rater", str(e))
            return
        self.rate(picture)

    def gender_changed(self):
        # Rate the same picture again against the other model face
        if self.mode == "result" and self.currentPicture is not None:
            self.rate(self.currentPicture)

    def rate(self, picture):
        self.mode = "result"
        self.currentPicture = picture
        self.backButton.pack(side="left", padx=4)
        self.takeButton.config(state="disabled")
        self.stop_animation()
        self.scoreLabel.config(text=" ")
        self.detailLabel.config(text="\n")
        self.show_picture(picture)
        self.set_status("Rating...", NEUTRAL)
        self.root.update_idletasks()

        try:
            result = ld.rate_face(picture, self.gender.get())
        except ld.FaceError as e:
            self.set_status(str(e), BAD)
            self.detailLabel.config(text="Take a new photo, or pick another one.\n")
            return

        self.show_picture(result["picture"])
        self.set_status("Done!", GOOD)

        clarity = result["clarity"]
        if clarity is None:
            skinText = "Could not see the cheeks, so skin clarity is not counted."
        else:
            skinText = f"Skin clarity on the cheeks: {round(clarity * 100)}%  (score x{result['skinFactor']:.2f})"
        self.detailLabel.config(text=skinText + "\nGreen dots: landmarks   Blue squares: cheeks checked   "
                                                 "Red: uneven skin")
        self.animate_score(result["score"])

    def animate_score(self, score, duration=1.5):
        # Count up to the score, like the terminal version did
        start = time.perf_counter()

        def step():
            t = min((time.perf_counter() - start) / duration, 1.0)
            shown = score * (1 - (1 - t) ** 3)  # slows down towards the end
            self.scoreLabel.config(text=f"Beauty score: {shown:,.1f}")
            self.animation = self.root.after(16, step) if t < 1.0 else None

        step()

    def stop_animation(self):
        if self.animation is not None:
            self.root.after_cancel(self.animation)
            self.animation = None

    def back_to_camera(self):
        if self.mode != "result":
            return
        self.stop_animation()
        self.mode = "camera"
        self.currentPicture = None
        self.backButton.pack_forget()
        self.scoreLabel.config(text=" ")
        self.detailLabel.config(text="\n")
        self.set_status("", NEUTRAL)

    # ---------- Helpers ----------

    def set_status(self, text, color):
        if self.status.cget("text") != text:
            self.status.config(text=text, fg=color)

    def show_picture(self, bgrPicture):
        # Scale the picture to fit the picture area, keeping its shape
        h, w = bgrPicture.shape[:2]
        shrink = min(self.viewW / w, self.viewH / h)
        size = (max(1, int(w * shrink)), max(1, int(h * shrink)))
        resized = cv2.resize(bgrPicture, size, interpolation=cv2.INTER_AREA if shrink < 1 else cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.view.config(image=self.photo, text="")


def main():
    if sys.platform == "win32":
        # Sharp text on high-DPI screens instead of blurry stretched text
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    FaceRaterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

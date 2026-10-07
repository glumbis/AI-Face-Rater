import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog, messagebox

try:
    import cv2
    from PIL import Image, ImageColor, ImageDraw, ImageTk
    import landmarkdetect as ld
except ImportError as e:
    # Started by double-clicking there's no terminal to show the error in, so show it in a window
    root = tk.Tk()
    root.withdraw()
    # e.name is None when the package was found but something inside it failed to import, then show the whole message
    messagebox.showerror("AI Face Rater", f"Missing Python package: {e.name or e}\n\n"
                                          "Install the packages with:\npip install -r requirements.txt")
    sys.exit(1)


# Which camera to use, 0 is the PC's default camera
CAMERA_INDEX = 0
# The live preview looks for the face in a copy this wide, which is much faster than the full picture
PREVIEW_DETECT_WIDTH = 320
# How often the live preview updates, in milliseconds
PREVIEW_DELAY = 30
# If the camera hasn't sent a new picture for this many seconds, it counts as disconnected
CAMERA_LOST_AFTER = 1.0
# How often (in milliseconds) the window checks if Windows switched between light and dark mode
THEME_CHECK_DELAY = 2000

# Sizes are written for 100% screen scaling and multiplied by the real scaling when the window is built
PAD = 24  # space around everything
PANEL_WIDTH = 300  # the column with the score and buttons
STATUS_ROW = 44  # the row under the picture that holds the status pill
WINDOW_FRAME = 48  # room for the title bar and window borders
MAX_VIEW_HEIGHT = 660
MIN_VIEW_HEIGHT = 300

# The status pill has one of these kinds, each with its own colours (see the colour themes below)
GOOD = "good"
WARN = "warn"
BAD = "bad"
NEUTRAL = "neutral"

# Colours. Each theme has the same names, so the rest of the code never needs to know which theme is used.
# Pills are (background, text colour). Only one accent colour is used, everything else is gray
LIGHT = {
    "bg": "#f3f3f3", "text": "#1a1a1a", "muted": "#666666", "faint": "#9c9c9c", "footer": "#8c8c8c",
    "stage": "#e3e3e3", "track": "#dcdcdc", "thumb": "#ffffff", "thumbEdge": "#d0d0d0",
    "trackEdge": "#dcdcdc", "segText": "#1a1a1a", "segOff": "#666666", "scoreHigh": "#0067c0", "scoreLow": "#b45309",
    "accent": "#0067c0", "accentHover": "#1a78cb", "accentPress": "#3d8ad2", "onAccent": "#ffffff",
    "button": "#ffffff", "buttonHover": "#f8f8f8", "buttonPress": "#ececec", "buttonEdge": "#d0d0d0",
    "off": "#e6e6e6", "offText": "#a0a0a0",
    GOOD: ("#dff3dc", "#0e5a0e"), WARN: ("#fff1c2", "#6b4700"),
    BAD: ("#fde4e6", "#a4262c"), NEUTRAL: ("#e6e6e6", "#5a5a5a"),
}
DARK = {
    "bg": "#202020", "text": "#f5f5f5", "muted": "#a3a3a3", "faint": "#6f6f6f", "footer": "#7d7d7d",
    "stage": "#161616", "track": "#1f1f1f", "thumb": "#5e5e5e", "thumbEdge": "#6b6b6b",
    "trackEdge": "#333333", "segText": "#ffffff", "segOff": "#b0b0b0", "scoreHigh": "#60cdff", "scoreLow": "#f3c969",
    "accent": "#60cdff", "accentHover": "#78d5ff", "accentPress": "#52b3e0", "onAccent": "#000000",
    "button": "#2e2e2e", "buttonHover": "#373737", "buttonPress": "#292929", "buttonEdge": "#3e3e3e",
    "off": "#2a2a2a", "offText": "#6d6d6d",
    GOOD: ("#1d3a23", "#6ccb5f"), WARN: ("#3d3417", "#f3d26b"),
    BAD: ("#4a2428", "#ff99a4"), NEUTRAL: ("#2e2e2e", "#b4b4b4"),
}

# What the score shows (faded) when there is no score yet. A dash can't be mistaken for a real score of 0
EMPTY_SCORE = "\u2013"
TIP_TEXT = "Look straight at the camera and keep a neutral face."
NO_CAMERA_TEXT = "Pick a photo to get started."
# The hint above the buttons is always this many lines high, so the window doesn't change size when it changes
HINT_LINES = 2
# The colours of the landmark dots and uneven spots drawn on the rated picture (the same as in landmarkdetect.py)
LEGEND_MINT = "#4ade80"
LEGEND_RED = "#f87171"
# How often (in milliseconds) the dots after "Loading..." move on
LOADING_DOTS_DELAY = 400

# How much bigger shapes are drawn before shrinking them again, which makes their edges smooth
SMOOTHING = 4


class Camera:
    # Reads frames from the camera in the background, so the window doesn't freeze while waiting for the next one

    def __init__(self, index):
        self.index = index
        self.frame = None
        self.frameTime = 0.0  # when the latest frame arrived
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
                self.frameTime = time.monotonic()
        cap.release()

    def lost(self):
        # True when the camera worked but has stopped sending pictures (for example unplugged).
        # Reading keeps failing for a few seconds before the loop above gives up, and until then the last
        # picture would stay on the screen as if nothing was wrong
        with self.lock:
            return self.frame is not None and time.monotonic() - self.frameTime > CAMERA_LOST_AFTER

    def latest(self):
        # The newest frame, or None if there isn't one yet or it's too old
        if self.lost():
            return None
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


# ---------- Drawing helpers ----------

def windows_uses_dark_mode():
    # Reads the "Choose your mode" setting from Windows. Anywhere else, or if it can't be read, use light
    if sys.platform != "win32":
        return False
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        with key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def shape_image(width, height, radius, fill, edge=None, dot=None):
    # A rounded rectangle with smooth edges and see-through corners, drawn as a picture because tkinter can't do this.
    # dot is (middle x, size, colour) for the small coloured circle in the status pill
    fillRgb = ImageColor.getrgb(fill)
    big = Image.new("RGBA", (width * SMOOTHING, height * SMOOTHING), fillRgb + (0,))
    draw = ImageDraw.Draw(big)
    draw.rounded_rectangle((0, 0, width * SMOOTHING - 1, height * SMOOTHING - 1), radius=radius * SMOOTHING,
                           fill=fillRgb + (255,), outline=edge, width=SMOOTHING)
    if dot is not None:
        middle, size, colour = dot
        top = (height - size) / 2
        draw.ellipse(((middle - size / 2) * SMOOTHING, top * SMOOTHING,
                      (middle + size / 2) * SMOOTHING, (top + size) * SMOOTHING), fill=colour)
    return big.resize((width, height), Image.Resampling.LANCZOS)


class RoundButton(tk.Label):
    # A flat rounded button. It's a label showing a picture of the button, because ttk buttons can't be rounded.
    # It never takes the keyboard focus, so Space can't press it by accident
    def __init__(self, parent, app, text, command, primary, width, height):
        font = app.fonts["buttonBold"] if primary else app.fonts["button"]
        super().__init__(parent, text=text, font=font, compound="center", bd=0, padx=0, pady=0,
                         highlightthickness=0, takefocus=0)
        self.app = app
        self.command = command
        self.primary = primary
        self.size = (width, height)
        self.enabled = True
        self.hover = False
        self.pressed = False
        self.images = {}
        self.bind("<Enter>", lambda e: self.set_hover(True))
        self.bind("<Leave>", lambda e: self.set_hover(False))
        self.bind("<ButtonPress-1>", self.press)
        self.bind("<ButtonRelease-1>", self.release)
        self.restyle()

    def restyle(self):
        c = self.app.colors
        width, height = self.size
        radius = self.app.px(8)
        if self.primary:
            fills = {"normal": c["accent"], "hover": c["accentHover"], "pressed": c["accentPress"]}
            edge = None
            self.textColors = {"normal": c["onAccent"], "hover": c["onAccent"], "pressed": c["onAccent"]}
        else:
            fills = {"normal": c["button"], "hover": c["buttonHover"], "pressed": c["buttonPress"]}
            edge = c["buttonEdge"]
            self.textColors = {"normal": c["text"], "hover": c["text"], "pressed": c["text"]}
        fills["disabled"] = c["off"]
        self.textColors["disabled"] = c["offText"]
        self.images = {state: ImageTk.PhotoImage(shape_image(width, height, radius, fill, None if state == "disabled" else edge))
                       for state, fill in fills.items()}
        # Swap to a new picture in the same call as the new colour, the old pictures are gone by now
        self.config(bg=c["bg"], image=self.images[self.state()])
        self.draw()

    def state(self):
        if not self.enabled:
            return "disabled"
        if self.pressed:
            return "pressed"
        return "hover" if self.hover else "normal"

    def draw(self):
        state = self.state()
        self.config(image=self.images[state], fg=self.textColors[state], cursor="hand2" if self.enabled else "")

    def set_enabled(self, enabled):
        if self.enabled != enabled:
            self.enabled = enabled
            self.draw()

    def set_text(self, text):
        self.config(text=text)

    def set_hover(self, hover):
        self.hover = hover
        if not hover:
            self.pressed = False
        self.draw()

    def press(self, event):
        self.pressed = True
        self.draw()

    def release(self, event):
        wasPressed = self.pressed
        self.pressed = False
        self.draw()
        # Only a click that also ends on the button counts, so you can slide away to cancel
        if wasPressed and self.enabled and 0 <= event.x < self.size[0] and 0 <= event.y < self.size[1]:
            self.command()


class Segmented(tk.Canvas):
    # Two or more choices side by side with a sliding highlight, like the switches in Windows 11
    def __init__(self, parent, app, variable, options, command, width, height):
        super().__init__(parent, width=width, height=height, bd=0, highlightthickness=0, takefocus=0)
        self.app = app
        self.variable = variable
        self.options = options  # list of (text, value)
        self.command = command
        self.size = (width, height)
        self.images = []
        self.bind("<Button-1>", self.click)
        # Redraw when the variable is changed from anywhere, not only by clicking here
        variable.trace_add("write", lambda *args: self.restyle())
        self.restyle()

    def restyle(self):
        c = self.app.colors
        width, height = self.size
        inset = self.app.px(3)
        radius = self.app.px(8)
        count = len(self.options)
        cellWidth = (width - 2 * inset) // count
        self.delete("all")
        self.config(bg=c["bg"])
        self.images = [ImageTk.PhotoImage(shape_image(width, height, radius, c["track"], c["trackEdge"]))]
        self.create_image(0, 0, anchor="nw", image=self.images[0])
        values = [value for _, value in self.options]
        selected = values.index(self.variable.get()) if self.variable.get() in values else 0
        thumb = ImageTk.PhotoImage(shape_image(cellWidth, height - 2 * inset, radius - inset // 2,
                                               c["thumb"], c["thumbEdge"]))
        self.images.append(thumb)
        self.create_image(inset + selected * cellWidth, inset, anchor="nw", image=thumb)
        for i, (text, value) in enumerate(self.options):
            self.create_text(inset + i * cellWidth + cellWidth / 2, height / 2, text=text, font=self.app.fonts["button"],
                             fill=c["segText"] if i == selected else c["segOff"])

    def click(self, event):
        inset = self.app.px(3)
        cellWidth = (self.size[0] - 2 * inset) // len(self.options)
        index = min(max((event.x - inset) // cellWidth, 0), len(self.options) - 1)
        value = self.options[index][1]
        if value != self.variable.get():
            self.variable.set(value)
            self.command()


class StatusPill(tk.Canvas):
    # A small rounded label with a coloured dot, for hints and errors. Empty text shows nothing but keeps its space
    def __init__(self, parent, app, width, height):
        super().__init__(parent, width=width, height=height, bd=0, highlightthickness=0, takefocus=0)
        self.app = app
        self.size = (width, height)
        self.text = ""
        self.kind = NEUTRAL
        self.image = None
        self.restyle()

    def show(self, text, kind):
        if (text, kind) != (self.text, self.kind):
            self.text = text
            self.kind = kind
            self.restyle()

    def restyle(self):
        c = self.app.colors
        width, height = self.size
        self.delete("all")
        self.config(bg=c["bg"])
        if not self.text:
            return
        font = self.app.fonts["small"]
        padding = self.app.px(14)
        dotSize = self.app.px(8)
        gap = self.app.px(8)
        pillHeight = self.app.px(32)
        # A long message is cut with "..." instead of making the pill wider than the picture
        text = self.text
        room = width - 2 * padding - dotSize - gap
        while font.measure(text) > room and len(text) > 1:
            text = text[:-2].rstrip() + "…"
        pillWidth = padding * 2 + dotSize + gap + font.measure(text)
        background, textColour = c[self.kind]
        self.image = ImageTk.PhotoImage(shape_image(pillWidth, pillHeight, pillHeight // 2, background,
                                                    dot=(padding + dotSize / 2, dotSize, textColour)))
        left = (width - pillWidth) // 2
        top = (height - pillHeight) // 2
        self.create_image(left, top, anchor="nw", image=self.image)
        self.create_text(left + padding + dotSize + gap, height // 2, text=text, anchor="w", font=font, fill=textColour)


class Bar(tk.Label):
    # A thin progress bar from 0 to 1, drawn as a picture so the ends are round
    def __init__(self, parent, app, width, height):
        super().__init__(parent, bd=0, padx=0, pady=0, highlightthickness=0, takefocus=0)
        self.app = app
        self.size = (width, height)
        self.value = None
        self.image = None
        self.restyle()

    def set_value(self, value):
        # value is 0 to 1, or None for an empty bar
        if value != self.value:
            self.value = value
            self.restyle()

    def restyle(self):
        c = self.app.colors
        width, height = self.size
        picture = shape_image(width, height, height // 2, c["track"])
        if self.value:
            fillWidth = max(height, round(width * min(self.value, 1)))
            picture.alpha_composite(shape_image(fillWidth, height, height // 2, c["accent"]))
        self.image = ImageTk.PhotoImage(picture)
        self.config(image=self.image, bg=c["bg"])


class Legend(tk.Canvas):
    # One small row that explains the colours drawn on the rated picture. It's empty until there is a result
    def __init__(self, parent, app, width, height):
        super().__init__(parent, width=width, height=height, bd=0, highlightthickness=0, takefocus=0)
        self.app = app
        self.size = (width, height)
        self.shown = False
        self.images = []
        self.restyle()

    def show(self, shown):
        if shown != self.shown:
            self.shown = shown
            self.restyle()

    def restyle(self):
        c = self.app.colors
        height = self.size[1]
        self.delete("all")
        self.config(bg=c["bg"])
        self.images = []
        if not self.shown:
            return
        size = self.app.px(8)
        font = self.app.fonts["tiny"]
        x = 0
        for kind, colour, text in (("dot", LEGEND_MINT, "Landmarks"), ("square", None, "Cheeks"),
                                   ("dot", LEGEND_RED, "Uneven skin")):
            if kind == "dot":
                swatch = shape_image(size, size, size // 2, colour)
            else:
                # The cheek squares are drawn as an outline, so their swatch is an outline too
                swatch = shape_image(size, size, self.app.px(2), c["bg"], edge=c["muted"])
            image = ImageTk.PhotoImage(swatch)
            self.images.append(image)
            self.create_image(x, (height - size) // 2, anchor="nw", image=image)
            x += size + self.app.px(6)
            self.create_text(x, height // 2, text=text, anchor="w", font=font, fill=c["muted"])
            x += font.measure(text) + self.app.px(16)


class FaceRaterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Rater")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        # The layout is fixed, so nothing moves around when the state changes
        self.root.resizable(False, False)

        self.camera = None
        self.mode = "camera"  # "camera" shows the live preview, "result" shows a rated picture
        self.currentPicture = None
        self.currentMirrored = False
        self.animation = None
        self.canTakePhoto = False
        self.scoreActive = False  # True while a score is shown
        self.scoreColour = "text"  # the name of the colour of the score, it depends on how good the score is
        self.cameraHint = TIP_TEXT  # what the hint says in camera mode
        self.dotCount = 3  # how many dots to show after "Loading"
        self.lastPicture = None  # what the big picture shows now (None = just a message), for redrawing on theme change
        self.message = None
        self.themed = []  # (widget, {option: colour name}), everything that changes colour with the theme
        self.photo = None

        # Windows' screen scaling (1.0 at 100%). tk scaling is pixels per point, which is 96/72 at 100%
        self.scale = float(self.root.tk.call("tk", "scaling")) / (96 / 72)
        self.dark = windows_uses_dark_mode()
        self.colors = DARK if self.dark else LIGHT
        self.make_fonts()

        # Picture area fits the screen (leaving room for the rest of the window), 4:3 like most webcams
        areaLeft, areaTop, areaWidth, areaHeight = self.work_area()
        maxHeight = areaHeight - self.px(WINDOW_FRAME + 2 * PAD + STATUS_ROW)
        maxWidth = areaWidth - self.px(WINDOW_FRAME + 3 * PAD + PANEL_WIDTH)
        self.viewH = max(self.px(MIN_VIEW_HEIGHT), min(self.px(MAX_VIEW_HEIGHT), maxHeight, maxWidth * 3 // 4))
        self.viewW = self.viewH * 4 // 3
        self.make_corner_mask()

        self.build_window()

        self.root.bind("<space>", self.space_pressed)
        self.root.bind("<Escape>", lambda e: self.back_to_camera())
        self.root.bind("<Control-o>", lambda e: self.pick_photo())

        self.apply_theme()
        self.show_message("Loading the face model...")
        self.reset_result_panel()
        self.refresh_buttons()

        # Put the window in the middle of the usable screen
        self.root.update_idletasks()
        x = areaLeft + max(0, (areaWidth - self.root.winfo_reqwidth()) // 2)
        y = areaTop + max(0, (areaHeight - self.root.winfo_reqheight() - self.px(WINDOW_FRAME) // 2) // 2)
        self.root.geometry(f"+{x}+{y}")

        self.root.after(50, self.start)
        self.root.after(THEME_CHECK_DELAY, self.check_theme)
        self.root.after(LOADING_DOTS_DELAY, self.move_loading_dots)

    # ---------- Building the window ----------

    def px(self, size):
        # A size at 100% screen scaling, made as big as it should be on this screen
        return round(size * self.scale)

    def work_area(self):
        # Where windows can go: (left, top, width, height) of the screen without the taskbar, in pixels
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                rect = wintypes.RECT()
                # 0x30 is SPI_GETWORKAREA
                if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rect), 0):
                    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
            except Exception:
                pass
        return 0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight() - self.px(48)

    def make_fonts(self):
        # Windows 11 has "Segoe UI Variable", older Windows has "Segoe UI". Tk cuts long font names at 31 letters
        have = set(tkFont.families(self.root))

        def pick(*names):
            return next((name for name in names if name in have), "Segoe UI")

        text = pick("Segoe UI Variable Text", "Segoe UI")
        textBold = pick("Segoe UI Variable Text Semibold", "Segoe UI Semibold", "Segoe UI")
        display = pick("Segoe UI Variable Display Semib", "Segoe UI Semibold", "Segoe UI")
        self.fonts = {
            "title": tkFont.Font(root=self.root, family=display, size=19),
            "score": tkFont.Font(root=self.root, family=display, size=44),
            "scoreUnit": tkFont.Font(root=self.root, family=display, size=18),
            "body": tkFont.Font(root=self.root, family=text, size=11),
            "bodyBold": tkFont.Font(root=self.root, family=textBold, size=11),
            "button": tkFont.Font(root=self.root, family=text, size=11),
            "buttonBold": tkFont.Font(root=self.root, family=textBold, size=11),
            "hint": tkFont.Font(root=self.root, family=text, size=11),
            "small": tkFont.Font(root=self.root, family=text, size=10),
            "tiny": tkFont.Font(root=self.root, family=text, size=9),
            "message": tkFont.Font(root=self.root, family=text, size=13),
        }

    def make_corner_mask(self):
        # Used to round the corners of the big picture: white where the picture is, black in the corners
        size = (self.viewW * SMOOTHING, self.viewH * SMOOTHING)
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=self.px(16) * SMOOTHING, fill=255)
        self.cornerMask = mask.resize((self.viewW, self.viewH), Image.Resampling.LANCZOS)

    def themed_widget(self, widget, **options):
        # Remembers which colour name each option of the widget uses, so the colours can change with the theme
        self.themed.append((widget, options))
        return widget

    def label(self, parent, text, font, colour="text", **options):
        label = tk.Label(parent, text=text, font=self.fonts[font], bd=0, padx=0, pady=0, takefocus=0, **options)
        return self.themed_widget(label, bg="bg", fg=colour)

    def build_window(self):
        px = self.px
        main = self.themed_widget(tk.Frame(self.root), bg="bg")
        main.pack(padx=px(PAD), pady=px(PAD))
        self.themed_widget(self.root, bg="bg")

        # Left: the picture, with the status pill under it
        left = self.themed_widget(tk.Frame(main), bg="bg")
        left.grid(row=0, column=0, sticky="n")
        # The picture has no border or padding, otherwise a few pixels of it are cut off at each side
        self.view = tk.Label(left, bd=0, padx=0, pady=0, highlightthickness=0, compound="center", justify="center",
                             font=self.fonts["message"], takefocus=0)
        self.themed_widget(self.view, bg="bg", fg="muted")
        self.view.pack()
        self.statusPill = StatusPill(left, self, self.viewW, px(STATUS_ROW))
        self.statusPill.pack()

        # Right: four groups (title and model face, score, stats, hint and buttons). The two in the middle share
        # the spare height, so the groups are spread out evenly
        right = self.themed_widget(tk.Frame(main), bg="bg")
        right.grid(row=0, column=1, sticky="ns", padx=(px(PAD), 0))
        # Rows 1, 3 and 5 are empty and share the spare height equally, so the gaps between the groups are equal
        for row in (1, 3, 5):
            right.rowconfigure(row, weight=1, minsize=px(12))
        panelW = px(PANEL_WIDTH)
        self.customWidgets = []

        def group(row, sticky):
            frame = self.themed_widget(tk.Frame(right), bg="bg")
            frame.grid(row=row, column=0, sticky=sticky)
            return frame

        header = group(0, "nw")
        self.label(header, "Face Rater", "title", anchor="w").pack(anchor="w")
        self.label(header, "Compare with the model face of a", "small", "muted", anchor="w").pack(anchor="w", pady=(px(18), px(8)))
        self.gender = tk.StringVar(value="boy")
        self.genderToggle = Segmented(header, self, self.gender, [("Boy", "boy"), ("Girl", "girl")],
                                      self.gender_changed, panelW, px(40))
        self.genderToggle.pack()
        self.customWidgets.append(self.genderToggle)

        scoreGroup = group(2, "w")
        self.label(scoreGroup, "Beauty score", "small", "muted", anchor="w").pack(anchor="w")
        scoreRow = self.themed_widget(tk.Frame(scoreGroup), bg="bg")
        scoreRow.pack(fill="x")
        # The digits of this font are all equally wide, so the "/ 10" next to the number stays in place while it counts
        self.scoreLabel = self.label(scoreRow, EMPTY_SCORE, "score", "faint")
        self.scoreLabel.pack(side="left")
        # The "/ 10" sits on the same baseline as the number: both labels end at the bottom, but the text of the
        # bigger font hangs lower below its baseline, so the small one is lifted by the difference
        lift = self.fonts["score"].metrics("descent") - self.fonts["scoreUnit"].metrics("descent")
        self.scoreUnit = self.label(scoreRow, "", "scoreUnit", "muted")
        self.scoreUnit.pack(side="left", anchor="s", padx=(px(10), 0), pady=(0, lift))

        statsGroup = group(4, "w")
        self.clarityRow = self.make_stat_row(statsGroup, "Skin clarity", panelW, 0)
        self.symmetryRow = self.make_stat_row(statsGroup, "Symmetry", panelW, px(16))
        self.legend = Legend(statsGroup, self, panelW, px(20))
        self.legend.pack(anchor="w", pady=(px(14), 0))
        self.customWidgets.append(self.legend)

        # The hint sits right above the buttons. It is always this many lines high, so nothing moves when it changes
        bottom = group(6, "sew")
        self.hint = tk.Label(bottom, text=TIP_TEXT, font=self.fonts["hint"], justify="left", anchor="sw", bd=0, padx=0, pady=0,
                             wraplength=panelW, height=HINT_LINES, takefocus=0)
        self.themed_widget(self.hint, bg="bg", fg="muted")
        self.hint.pack(fill="x", pady=(0, px(16)))
        self.takeButton = RoundButton(bottom, self, "Take photo", self.primary_clicked, True, panelW, px(48))
        self.takeButton.pack()
        self.pickButton = RoundButton(bottom, self, "Pick a photo…", self.pick_photo, False, panelW, px(44))
        self.pickButton.pack(pady=(px(10), 0))
        self.pickButton.set_enabled(False)
        self.customWidgets += [self.takeButton, self.pickButton]
        self.label(bottom, "Space takes a photo  ·  Esc goes back", "tiny", "footer").pack(pady=(px(12), 0))

    def make_stat_row(self, parent, name, panelW, gap):
        # A name on the left, the percentage on the right and a bar under them
        frame = self.themed_widget(tk.Frame(parent), bg="bg")
        frame.pack(fill="x", pady=(gap, 0))
        frame.columnconfigure(0, weight=1)
        nameLabel = self.label(frame, name, "body", "text", anchor="w")
        nameLabel.grid(row=0, column=0, sticky="w")
        valueLabel = self.label(frame, "–", "bodyBold", "muted", anchor="e")
        valueLabel.grid(row=0, column=1, sticky="e")
        bar = Bar(frame, self, panelW, self.px(6))
        bar.grid(row=1, column=0, columnspan=2, sticky="w", pady=(self.px(8), 0))
        self.customWidgets.append(bar)
        return {"value": valueLabel, "bar": bar}

    # ---------- Colours ----------

    def apply_theme(self):
        # Gives every widget the colours of the current light or dark theme
        c = self.colors
        for widget, options in self.themed:
            widget.config(**{option: c[name] for option, name in options.items()})
        for widget in self.customWidgets:
            widget.restyle()
        self.statusPill.restyle()
        self.color_score(self.scoreActive, self.scoreColour)
        if self.lastPicture is not None:
            self.show_picture(self.lastPicture)
        else:
            self.redraw_message()
        self.style_title_bar()

    def check_theme(self):
        dark = windows_uses_dark_mode()
        if dark != self.dark:
            self.dark = dark
            self.colors = DARK if dark else LIGHT
            self.apply_theme()
        self.root.after(THEME_CHECK_DELAY, self.check_theme)

    def style_title_bar(self):
        # Makes the title bar match the theme (dark mode needs Windows 10 version 2004 or newer, the colour
        # needs Windows 11). On anything older Windows just ignores it
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.root.update_idletasks()
            window = ctypes.windll.user32.GetParent(self.root.winfo_id())
            dark = ctypes.c_int(1 if self.dark else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(window, 20, ctypes.byref(dark), 4)
            red, green, blue = ImageColor.getrgb(self.colors["bg"])
            colour = ctypes.c_int(red | green << 8 | blue << 16)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(window, 35, ctypes.byref(colour), 4)
        except Exception:
            pass

    # ---------- Starting and closing ----------

    def start(self):
        try:
            ld.load_predictor()
        except (FileNotFoundError, RuntimeError) as e:
            # RuntimeError is a damaged model file
            messagebox.showerror("AI Face Rater", str(e))
            self.root.destroy()
            return
        self.pickButton.set_enabled(True)
        self.show_message("Starting the camera...")
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
                # The message in the picture says what is wrong, so the status pill stays empty
                self.show_message("No camera found.")
                self.set_camera_hint(NO_CAMERA_TEXT)
                self.set_can_take_photo(False)
                self.set_status("", NEUTRAL)
            elif self.camera.lost():
                self.show_message("Camera disconnected.\nPlug it back in to continue.")
                self.set_camera_hint(NO_CAMERA_TEXT)
                self.set_can_take_photo(False)
                self.set_status("", NEUTRAL)
            else:
                frame = self.camera.latest()
                if frame is not None:
                    self.show_preview_frame(frame)
        self.root.after(PREVIEW_DELAY, self.update_preview)

    def show_preview_frame(self, frame):
        # Mirror it, so moving your head to the left moves it to the left on the screen, like a mirror
        frame = cv2.flip(frame, 1)
        self.set_can_take_photo(True)
        self.set_camera_hint(TIP_TEXT)

        h, w = frame.shape[:2]
        found = ld.landmark_detect(frame, detectScale=min(1.0, PREVIEW_DETECT_WIDTH / w))
        if found is None:
            self.set_status("No face found. Face the camera!", WARN)
        else:
            xList, yList = found
            problem = ld.facing_problem(xList, yList, frame.shape)
            if problem is None:
                self.set_status("Looking good! Press Take photo or Space.", GOOD)
                color = (128, 222, 74)
            else:
                self.set_status(f"Face the camera! You are facing {problem}.", WARN)
                color = (40, 160, 245)
            for x, y in zip(xList, yList):
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)

        self.show_picture(frame)

    # ---------- Rating ----------

    def take_photo(self):
        if self.mode != "camera" or self.camera is None or not self.camera.opened:
            return
        frame = self.camera.latest()
        if frame is None:
            return
        # Rate the picture the way the camera took it, like a photo picked from a file (the face is the
        # same way round as other people see it). It's only shown mirrored, like the preview
        self.rate(frame, mirrored=True)

    def primary_clicked(self):
        # The big button takes a photo, or after a result it goes back to the camera
        if self.mode == "result":
            self.back_to_camera()
        else:
            self.take_photo()

    def pick_photo(self):
        if not self.pickButton.enabled:
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
            self.rate(self.currentPicture, self.currentMirrored)

    def rate(self, picture, mirrored=False):
        # mirrored=True shows the picture (and the result) mirrored, but rates it as it is
        self.mode = "result"
        self.currentPicture = picture
        self.currentMirrored = mirrored
        self.refresh_buttons()
        self.stop_animation()
        self.reset_result_panel()
        self.show_picture(cv2.flip(picture, 1) if mirrored else picture)
        self.set_status("Rating...", NEUTRAL)
        self.root.update_idletasks()

        try:
            result = ld.rate_face(picture, self.gender.get())
        except ld.FaceError as e:
            self.set_status(str(e), BAD)
            self.set_hint("Take a new photo, or pick another one.")
            return

        self.show_picture(cv2.flip(result["picture"], 1) if mirrored else result["picture"])
        self.set_status(f"Rated against the {self.gender.get()} model", NEUTRAL)

        clarity = result["clarity"]
        if clarity is None:
            self.clarityRow["value"].config(text="Not counted", fg=self.colors["muted"])
            self.clarityRow["bar"].set_value(None)
            self.set_hint("Skin clarity is not counted: the cheeks are hidden, bearded or black-and-white.")
        else:
            self.clarityRow["value"].config(text=f"{round(clarity * 100)}%", fg=self.colors["text"])
            self.clarityRow["bar"].set_value(clarity)
        self.symmetryRow["value"].config(text=f"{round(result['symmetry'] * 100)}%", fg=self.colors["text"])
        self.symmetryRow["bar"].set_value(result["symmetry"])
        self.legend.show(True)
        self.animate_score(result["score"])

    def animate_score(self, score, duration=1.5):
        # Count up to the score, like the terminal version did
        start = time.perf_counter()
        # The number is tinted by how good the score is: blue for high, amber for low, plain in between
        self.color_score(True, "scoreHigh" if score >= 8 else "scoreLow" if score < 4 else "text")

        def step():
            t = min((time.perf_counter() - start) / duration, 1.0)
            shown = score * (1 - (1 - t) ** 3)  # slows down towards the end
            self.scoreLabel.config(text=f"{shown:.1f}")
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
        self.refresh_buttons()
        self.reset_result_panel()
        self.set_status("", NEUTRAL)

    # ---------- Score panel ----------

    def color_score(self, active, colour="text"):
        # The score has its own colour when there is one, and is faded (with no "/ 10") while there isn't
        self.scoreActive = active
        self.scoreColour = colour
        self.scoreLabel.config(fg=self.colors[colour] if active else self.colors["faint"])
        self.scoreUnit.config(text="/ 10" if active else "", fg=self.colors["muted"])

    def reset_result_panel(self):
        # Empty score and stats still take up their space, so the window doesn't change size when a result shows up
        self.scoreLabel.config(text=EMPTY_SCORE)
        self.color_score(False)
        for row in (self.clarityRow, self.symmetryRow):
            row["value"].config(text="–", fg=self.colors["muted"])
            row["bar"].set_value(None)
        self.legend.show(False)
        self.set_hint(self.cameraHint if self.mode == "camera" else "")

    def set_hint(self, text):
        if self.hint.cget("text") != text:
            self.hint.config(text=text)

    def set_camera_hint(self, text):
        # The hint for camera mode. A result has its own hints, so it only shows when the camera is on
        self.cameraHint = text
        if self.mode == "camera":
            self.set_hint(text)

    def refresh_buttons(self):
        if self.mode == "result":
            self.takeButton.set_text("Back to camera")
            self.takeButton.set_enabled(True)
        else:
            self.takeButton.set_text("Take photo")
            self.takeButton.set_enabled(self.canTakePhoto)

    def set_can_take_photo(self, can):
        if can != self.canTakePhoto:
            self.canTakePhoto = can
            self.refresh_buttons()

    # ---------- Helpers ----------

    def space_pressed(self, event):
        self.take_photo()
        # Stop here, so Space doesn't also do anything else
        return "break"

    def set_status(self, text, kind):
        self.statusPill.show(text, kind)

    def stage(self, bgrPicture):
        # The big picture area as a PIL image: the picture scaled to fit (keeping its shape) on a calm background,
        # with rounded corners
        stage = Image.new("RGB", (self.viewW, self.viewH), self.colors["stage"])
        if bgrPicture is not None:
            h, w = bgrPicture.shape[:2]
            shrink = min(self.viewW / w, self.viewH / h)
            size = (max(1, int(w * shrink)), max(1, int(h * shrink)))
            resized = cv2.resize(bgrPicture, size, interpolation=cv2.INTER_AREA if shrink < 1 else cv2.INTER_LINEAR)
            stage.paste(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
                        ((self.viewW - size[0]) // 2, (self.viewH - size[1]) // 2))
        # The corners get the window's colour
        corners = Image.new("RGB", (self.viewW, self.viewH), self.colors["bg"])
        return Image.composite(stage, corners, self.cornerMask)

    def show_picture(self, bgrPicture):
        self.lastPicture = bgrPicture
        self.message = None
        self.photo = ImageTk.PhotoImage(self.stage(bgrPicture))
        self.view.config(image=self.photo, text="")

    def show_message(self, text):
        # Text in the picture area instead of a picture ("Loading...", "No camera found.").
        # A message ending in "..." gets moving dots, so it's clear the app is working
        if self.message == text and self.lastPicture is None:
            return
        self.lastPicture = None
        self.message = text
        self.redraw_message()

    def message_text(self):
        if self.message and self.message.endswith("..."):
            # Spaces fill up the missing dots, so the text stays in the same place
            return self.message[:-3] + "." * self.dotCount + " " * (3 - self.dotCount)
        return self.message or ""

    def redraw_message(self):
        self.photo = ImageTk.PhotoImage(self.stage(None))
        self.view.config(image=self.photo, text=self.message_text())

    def move_loading_dots(self):
        if self.message and self.message.endswith("...") and self.lastPicture is None:
            self.dotCount = (self.dotCount + 1) % 4
            self.view.config(text=self.message_text())
        self.root.after(LOADING_DOTS_DELAY, self.move_loading_dots)


def main():
    if sys.platform == "win32":
        # Sharp text on high-DPI screens instead of blurry stretched text
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()

    def show_error(excType, value, trace):
        # Started with pythonw there's nowhere for errors to be printed, so show them instead of failing silently
        import traceback
        traceback.print_exception(excType, value, trace)
        messagebox.showerror("AI Face Rater", f"Something went wrong:\n\n{excType.__name__}: {value}")

    root.report_callback_exception = show_error
    FaceRaterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

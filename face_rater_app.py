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
    "bg": "#f3f3f3", "text": "#1a1a1a", "muted": "#666666", "faint": "#b8b8b8",
    "stage": "#e3e3e3", "track": "#dcdcdc", "thumb": "#ffffff", "thumbEdge": "#d0d0d0",
    "accent": "#0067c0", "accentHover": "#1a78cb", "accentPress": "#3d8ad2", "onAccent": "#ffffff",
    "button": "#ffffff", "buttonHover": "#f8f8f8", "buttonPress": "#ececec", "buttonEdge": "#d6d6d6",
    "off": "#e6e6e6", "offText": "#a0a0a0",
    GOOD: ("#dff3dc", "#0e5a0e"), WARN: ("#fff1c2", "#6b4700"),
    BAD: ("#fde4e6", "#a4262c"), NEUTRAL: ("#e6e6e6", "#5a5a5a"),
}
DARK = {
    "bg": "#202020", "text": "#f5f5f5", "muted": "#a3a3a3", "faint": "#474747",
    "stage": "#161616", "track": "#2f2f2f", "thumb": "#4b4b4b", "thumbEdge": "#5a5a5a",
    "accent": "#60cdff", "accentHover": "#78d5ff", "accentPress": "#52b3e0", "onAccent": "#000000",
    "button": "#2e2e2e", "buttonHover": "#373737", "buttonPress": "#292929", "buttonEdge": "#3e3e3e",
    "off": "#2a2a2a", "offText": "#6d6d6d",
    GOOD: ("#1d3a23", "#6ccb5f"), WARN: ("#3d3417", "#f3d26b"),
    BAD: ("#4a2428", "#ff99a4"), NEUTRAL: ("#2e2e2e", "#b4b4b4"),
}

# What the score shows (faded) when there is no score yet, so the space is already taken
EMPTY_SCORE = "0.0"
TIP_TEXT = "Look straight at the camera and keep a neutral face."
LEGEND_TEXT = "Green dots: landmarks\nBlue squares: cheeks checked\nRed: uneven skin"
# The text under the stats is always this many lines high, so the window doesn't change size when it changes
CAPTION_LINES = 4

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
        self.images = [ImageTk.PhotoImage(shape_image(width, height, radius, c["track"]))]
        self.create_image(0, 0, anchor="nw", image=self.images[0])
        values = [value for _, value in self.options]
        selected = values.index(self.variable.get()) if self.variable.get() in values else 0
        thumb = ImageTk.PhotoImage(shape_image(cellWidth, height - 2 * inset, radius - inset // 2,
                                               c["thumb"], c["thumbEdge"]))
        self.images.append(thumb)
        self.create_image(inset + selected * cellWidth, inset, anchor="nw", image=thumb)
        for i, (text, value) in enumerate(self.options):
            self.create_text(inset + i * cellWidth + cellWidth / 2, height / 2, text=text, font=self.app.fonts["button"],
                             fill=c["text"] if i == selected else c["muted"])

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
        displayLight = pick("Segoe UI Variable Display Semil", "Segoe UI Semilight", "Segoe UI")
        self.fonts = {
            "title": tkFont.Font(root=self.root, family=display, size=19),
            "score": tkFont.Font(root=self.root, family=displayLight, size=52),
            "scoreUnit": tkFont.Font(root=self.root, family=display, size=18),
            "body": tkFont.Font(root=self.root, family=text, size=11),
            "bodyBold": tkFont.Font(root=self.root, family=textBold, size=11),
            "button": tkFont.Font(root=self.root, family=text, size=11),
            "buttonBold": tkFont.Font(root=self.root, family=textBold, size=11),
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

        # Right: title, model face choice, score and buttons
        right = self.themed_widget(tk.Frame(main), bg="bg")
        right.grid(row=0, column=1, sticky="ns", padx=(px(PAD), 0))
        panelW = px(PANEL_WIDTH)

        # The buttons are packed first so they stick to the bottom, the rest fills from the top
        self.hintLabel = self.label(right, "Space takes a photo  ·  Esc goes back", "tiny", "muted")
        self.hintLabel.pack(side="bottom", pady=(px(10), 0))
        self.pickButton = RoundButton(right, self, "Pick a photo…", self.pick_photo, False, panelW, px(44))
        self.pickButton.pack(side="bottom", pady=(px(10), 0))
        self.pickButton.set_enabled(False)
        self.takeButton = RoundButton(right, self, "Take photo", self.primary_clicked, True, panelW, px(48))
        self.takeButton.pack(side="bottom")
        self.customWidgets = [self.pickButton, self.takeButton]

        self.label(right, "Face Rater", "title", anchor="w").pack(anchor="w")

        self.label(right, "Compare with the model face of a", "small", "muted", anchor="w").pack(anchor="w", pady=(px(18), px(8)))
        self.gender = tk.StringVar(value="boy")
        self.genderToggle = Segmented(right, self, self.gender, [("Boy", "boy"), ("Girl", "girl")],
                                      self.gender_changed, panelW, px(40))
        self.genderToggle.pack()
        self.customWidgets.append(self.genderToggle)

        self.label(right, "Beauty score", "small", "muted", anchor="w").pack(anchor="w", pady=(px(22), 0))
        scoreRow = self.themed_widget(tk.Frame(right), bg="bg")
        scoreRow.pack(fill="x")
        # The digits of this font are all equally wide, so the "/ 10" next to the number stays in place while it counts
        self.scoreLabel = self.label(scoreRow, EMPTY_SCORE, "score", "faint")
        self.scoreLabel.pack(side="left")
        self.scoreUnit = self.label(scoreRow, "/ 10", "scoreUnit", "faint")
        self.scoreUnit.pack(side="left", anchor="s", padx=(self.px(10), 0), pady=(0, self.px(12)))

        self.clarityRow = self.make_stat_row(right, "Skin clarity", panelW)
        self.symmetryRow = self.make_stat_row(right, "Symmetry", panelW)

        # The text under the stats, always this many lines high
        self.caption = tk.Label(right, text="", font=self.fonts["tiny"], justify="left", anchor="nw", bd=0, padx=0, pady=0,
                                wraplength=panelW, height=CAPTION_LINES, takefocus=0)
        self.themed_widget(self.caption, bg="bg", fg="muted")
        self.caption.pack(fill="x", pady=(px(14), 0))

    def make_stat_row(self, parent, name, panelW):
        # A name on the left, the percentage on the right and a bar under them
        frame = self.themed_widget(tk.Frame(parent), bg="bg")
        frame.pack(fill="x", pady=(self.px(16), 0))
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
        self.color_score(self.scoreActive)
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
                self.show_message("No camera found.\nUse \"Pick a photo…\" instead.")
                self.set_can_take_photo(False)
                self.set_status("", NEUTRAL)
            elif self.camera.lost():
                self.show_message("Camera disconnected.\nPlug it back in, or use \"Pick a photo…\".")
                self.set_can_take_photo(False)
                self.set_status("Camera disconnected.", BAD)
            else:
                frame = self.camera.latest()
                if frame is not None:
                    self.show_preview_frame(frame)
        self.root.after(PREVIEW_DELAY, self.update_preview)

    def show_preview_frame(self, frame):
        # Mirror it, so moving your head to the left moves it to the left on the screen, like a mirror
        frame = cv2.flip(frame, 1)
        self.set_can_take_photo(True)

        h, w = frame.shape[:2]
        found = ld.landmark_detect(frame, detectScale=min(1.0, PREVIEW_DETECT_WIDTH / w))
        if found is None:
            self.set_status("No face found. Face the camera!", WARN)
        else:
            xList, yList = found
            problem = ld.facing_problem(xList, yList, frame.shape)
            if problem is None:
                self.set_status("Looking good! Press Take photo or Space.", GOOD)
                color = (80, 200, 80)
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
            self.caption.config(text="Take a new photo, or pick another one.")
            return

        self.show_picture(cv2.flip(result["picture"], 1) if mirrored else result["picture"])
        self.set_status("Done!", GOOD)

        clarity = result["clarity"]
        if clarity is None:
            self.clarityRow["value"].config(text="Not counted", fg=self.colors["muted"])
            self.clarityRow["bar"].set_value(None)
            note = "\n\nCheeks hidden, bearded or black-and-white, so skin clarity is not counted."
        else:
            self.clarityRow["value"].config(text=f"{round(clarity * 100)}%", fg=self.colors["text"])
            self.clarityRow["bar"].set_value(clarity)
            note = ""
        self.symmetryRow["value"].config(text=f"{round(result['symmetry'] * 100)}%", fg=self.colors["text"])
        self.symmetryRow["bar"].set_value(result["symmetry"])
        self.caption.config(text=LEGEND_TEXT + note)
        self.animate_score(result["score"])

    def animate_score(self, score, duration=1.5):
        # Count up to the score, like the terminal version did
        start = time.perf_counter()
        self.color_score(True)

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

    def color_score(self, active):
        # The score is dark when there is one, and faded while there isn't
        self.scoreActive = active
        self.scoreLabel.config(fg=self.colors["text"] if active else self.colors["faint"])
        self.scoreUnit.config(fg=self.colors["muted"] if active else self.colors["faint"])

    def reset_result_panel(self):
        # Empty score and stats still take up their space, so the window doesn't change size when a result shows up
        self.scoreLabel.config(text=EMPTY_SCORE)
        self.color_score(False)
        for row in (self.clarityRow, self.symmetryRow):
            row["value"].config(text="–", fg=self.colors["muted"])
            row["bar"].set_value(None)
        self.caption.config(text=TIP_TEXT if self.mode == "camera" else "")

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
        # Text in the picture area instead of a picture ("Loading...", "No camera found.")
        if self.message == text and self.lastPicture is None:
            return
        self.lastPicture = None
        self.message = text
        self.redraw_message()

    def redraw_message(self):
        self.photo = ImageTk.PhotoImage(self.stage(None))
        self.view.config(image=self.photo, text=self.message or "")


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

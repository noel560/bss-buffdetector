import cv2
import numpy as np
import mss
import time
import pyttsx3
import threading
import tkinter as tk
from tkinter import font

# -----------------------------
# SETTINGS
# -----------------------------
DEBUG = False

TARGET_COLOR = np.array([250, 41, 148])
COLOR_TOLERANCE = 20

PIXEL_COUNT_THRESHOLD = 80

FRAME_CONFIRM = 10
PICKUP_COOLDOWN = 1.5

timer_end = 0
token_frames = 0
last_pickup_time = 0

# -----------------------------
# TTS
# -----------------------------
tts_active = False
tts_engine = None
tts_thread = None

def tts_loop():
    global tts_active, tts_engine

    if tts_engine is None:
        tts_engine = pyttsx3.init('sapi5')
        tts_engine.setProperty('rate', 160)
        tts_engine.setProperty('volume', 0.9)

    tts_engine.startLoop(False)

    while tts_active:
        tts_engine.say("Precision")
        print("Refresh precision!")

        while tts_active and tts_engine.isBusy():
            tts_engine.iterate()
            time.sleep(0.01)

        time.sleep(1.5)

    if tts_engine:
        tts_engine.endLoop()
        tts_engine.stop()

# -----------------------------
# TKINTER WINDOW
# -----------------------------
root = tk.Tk()
root.iconbitmap("icon.ico")
root.title("BUFF DETECTOR")
root.attributes("-topmost", True)
root.geometry("340x130+50+50")
root.resizable(False, False)
root.configure(bg="#202020")

root.wm_attributes("-transparentcolor", "black")

main_frame = tk.Frame(root, bg="#202020", padx=16, pady=12)
main_frame.pack(fill="both", expand=True)

title_font = font.Font(family="Segoe UI", size=14, weight="bold")
title_label = tk.Label(
    main_frame,
    text="BUFF DETECTOR",
    font=title_font,
    fg="#f3f4f6",
    bg="#202020"
)
title_label.pack(anchor="n", pady=(0, 12))

separator = tk.Frame(main_frame, height=1, bg="#3a3a3a")
separator.pack(fill="x", pady=(0, 12))

content_frame = tk.Frame(main_frame, bg="#202020")
content_frame.pack(fill="x")

try:
    icon_img = tk.PhotoImage(file="precision_icon.png")
    icon_img = icon_img.zoom(1, 1).subsample(1, 1)
    icon_label = tk.Label(content_frame, image=icon_img, bg="#202020")
    icon_label.image = icon_img
    icon_label.pack(side="left", padx=(0, 14))
except Exception as e:
    print("precision.png hiba:", e)
    icon_label = tk.Label(
        content_frame,
        text="✦",
        font=("Segoe UI", 32, "bold"),
        fg="#9333ea",
        bg="#202020"
    )
    icon_label.pack(side="left", padx=(0, 14))

text_font = font.Font(family="Segoe UI", size=18, weight="bold")
label = tk.Label(
    content_frame,
    text="Precision: Off",
    font=text_font,
    fg="#99a1af",
    bg="#202020",
    anchor="w",
    justify="left"
)
label.pack(side="left", fill="x", expand=True)

# -----------------------------
# ROI
# -----------------------------
# token_roi = {
#     "top": 300,
#     "left": 800,
#     "width": 300,
#     "height": 300
# }
token_roi = {
    "top": 0,
    "left": 0,
    "width": 1920,
    "height": 1080
}


# -----------------------------
# DETECTION LOOP
# -----------------------------
def detection_loop():
    global timer_end, token_frames, last_pickup_time, tts_active

    lower = np.clip(TARGET_COLOR - COLOR_TOLERANCE, 0, 255)
    upper = np.clip(TARGET_COLOR + COLOR_TOLERANCE, 0, 255)

    tts_thread = None

    with mss.mss() as sct:
        while True:

            now = time.time()

            raw = np.array(sct.grab(token_roi))
            img = raw[:, :, :3]

            mask = cv2.inRange(img, lower, upper)
            pixel_count = cv2.countNonZero(mask)

            token_found = pixel_count > PIXEL_COUNT_THRESHOLD

            if token_found:
                token_frames += 1
            else:
                token_frames = 0

            if (
                token_frames >= FRAME_CONFIRM
                and now - last_pickup_time > PICKUP_COOLDOWN
            ):
                timer_end = now + 60
                last_pickup_time = now
                tts_active = False
                print(f"Precision pickup → reset ({pixel_count}px)")

            # -----------------
            # UI + TTS
            # -----------------
            if timer_end > now:
                remaining = timer_end - now

                if remaining > 20:
                    label.config(
                        text=f"Precision: {int(remaining)}s",
                        fg="#059669"
                    )
                    tts_active = False

                else:
                    label.config(
                        text="Precision: Refresh!",
                        fg="#64748b"
                    )

                    if not tts_active:
                        tts_active = True
                        if tts_thread is None or not tts_thread.is_alive():
                            tts_thread = threading.Thread(
                                target=tts_loop,
                                daemon=True
                            )
                            tts_thread.start()

            else:
                label.config(
                    text="Precision: Off",
                    fg="#99a1af"
                )
                tts_active = False

            # DEBUG PREVIEW
            if DEBUG:
                preview = img.copy()

                cv2.putText(preview, f"px:{pixel_count}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(preview, f"frames:{token_frames}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if token_found:
                    cv2.putText(preview, "DETECTED", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Token ROI", preview)
                cv2.imshow("Color Mask", mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)

# -----------------------------
# THREAD START
# -----------------------------
threading.Thread(
    target=detection_loop,
    daemon=True
).start()

root.mainloop()
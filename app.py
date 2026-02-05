import cv2
import numpy as np
import mss
import time
import pyttsx3
import threading
import tkinter as tk

# -----------------------------
# SETTINGS
# -----------------------------

DEBUG = True

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
root.title("Buff Detector")
root.attributes("-topmost", True)
root.geometry("220x80+50+50")
root.resizable(False, False)

label = tk.Label(
    root,
    text="Precision: Off",
    font=("Segoe UI", 16),
    fg="black",
    bg="white"
)
label.pack(fill="both", expand=True)

# -----------------------------
# ROI
# -----------------------------

token_roi = {
    "top": 300,
    "left": 800,
    "width": 300,
    "height": 300
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

            # Screenshot
            raw = np.array(sct.grab(token_roi))
            img = raw[:, :, :3]

            # -----------------
            # PIXEL MASK
            # -----------------
            mask = cv2.inRange(img, lower, upper)
            pixel_count = cv2.countNonZero(mask)

            token_found = pixel_count > PIXEL_COUNT_THRESHOLD

            # -----------------
            # FRAME CONFIRM
            # -----------------
            if token_found:
                token_frames += 1
            else:
                token_frames = 0

            # -----------------
            # TIMER RESET
            # -----------------
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
                        fg="black"
                    )
                    tts_active = False

                else:
                    label.config(
                        text="Precision: Refresh",
                        fg="red"
                    )

                    # TTS
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
                    fg="gray"
                )
                tts_active = False

            # -----------------
            # DEBUG PREVIEW
            # -----------------
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
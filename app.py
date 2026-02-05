import cv2
import numpy as np
import mss
import time
import pyttsx3
import threading
import tkinter as tk

# -----------------------------
# BEÁLLÍTÁSOK
# -----------------------------

DEBUG = True

# Precision szem színe (#9429FA → BGR)
TARGET_COLOR = np.array([250, 41, 148])
COLOR_TOLERANCE = 20

# Minimum pixel cluster
PIXEL_COUNT_THRESHOLD = 80

# Pickup védelem
FRAME_CONFIRM = 10
PICKUP_COOLDOWN = 1.5

timer_end = 0
token_frames = 0
last_pickup_time = 0

# -----------------------------
# TTS
# -----------------------------

tts_active = False

def tts_loop():
    global tts_active

    engine = pyttsx3.init()
    # engine.setProperty('rate', 180)   # opcionális: ha gyorsabb/lassabb beszéd kell
    # engine.setProperty('volume', 0.9)

    while tts_active:
        engine.say("Precision")
        engine.runAndWait()

        if not tts_active:
            break

        time.sleep(2.5)   # kb. 2-3 mp-ként mondja újra

# -----------------------------
# TKINTER ABLAK
# -----------------------------

root = tk.Tk()
root.title("Precision Timer")
root.attributes("-topmost", True)
root.geometry("220x80+50+50")
root.resizable(False, False)

label = tk.Label(
    root,
    text="Precision: Nincs",
    font=("Segoe UI", 16),
    fg="white",
    bg="black"
)
label.pack(fill="both", expand=True)

# -----------------------------
# ROI (SZEM HELYE)
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

    # Lokális változó a TTS thread kezelésére – NEM globális!
    tts_thread = None

    with mss.mss() as sct:
        while True:

            now = time.time()

            # Screenshot (BGRA → BGR)
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
                        fg="white"
                    )
                    tts_active = False

                else:
                    label.config(
                        text="Precision: Refresh",
                        fg="red"
                    )

                    # TTS indítása, ha még nem fut
                    if not tts_active:
                        tts_active = True
                        # Ha nincs thread VAGY már befejeződött → újraindítjuk
                        if tts_thread is None or not tts_thread.is_alive():
                            tts_thread = threading.Thread(
                                target=tts_loop,
                                daemon=True
                            )
                            tts_thread.start()

            else:
                label.config(
                    text="Precision: Nincs",
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
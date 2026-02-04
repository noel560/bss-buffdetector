import cv2
import numpy as np
import mss
import time
import pyttsx3
import threading
import tkinter as tk

# -----------------------------
# TEMPLATEK
# -----------------------------

buff_templates = [
    cv2.imread(f"new{i}.png", 0) for i in range(1, 11)
]

token_templates = [
    cv2.imread(f"par{i}.png", 0) for i in range(1, 6)
]

# -----------------------------
# BEÁLLÍTÁSOK
# -----------------------------

BUFF_THRESHOLD = 0.75
TOKEN_THRESHOLD = 0.7

timer_end = 0
token_last = False

# TTS
engine = pyttsx3.init()
tts_active = False

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
# ROI-K
# -----------------------------

buff_roi = {
    "top": 40,
    "left": 0,
    "width": 1920,
    "height": 100
}

token_roi = {
    "top": 300,
    "left": 800,
    "width": 300,
    "height": 300
}

# -----------------------------
# TTS THREAD
# -----------------------------

def tts_loop():
    global tts_active
    while tts_active:
        engine.say("Precision")
        engine.runAndWait()
        time.sleep(2.5)

# -----------------------------
# DETECTION LOOP THREAD
# -----------------------------

def detection_loop():
    global timer_end, token_last, tts_active

    with mss.mss() as sct:

        while True:
            now = time.time()

            buff_img = np.array(sct.grab(buff_roi))
            token_img = np.array(sct.grab(token_roi))

            buff_gray = cv2.cvtColor(buff_img, cv2.COLOR_BGR2GRAY)
            token_gray = cv2.cvtColor(token_img, cv2.COLOR_BGR2GRAY)

            # -----------------
            # BUFF DETECT
            # -----------------

            buff_found = False

            for template in buff_templates:
                if template is None:
                    continue

                res = cv2.matchTemplate(
                    buff_gray,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )

                if np.max(res) >= BUFF_THRESHOLD:
                    buff_found = True
                    break

            # -----------------
            # TOKEN DETECT
            # -----------------

            token_found = False

            for template in token_templates:
                if template is None:
                    continue

                res = cv2.matchTemplate(
                    token_gray,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )

                if np.max(res) >= TOKEN_THRESHOLD:
                    token_found = True
                    break

            # -----------------
            # TIMER LOGIKA
            # -----------------

            if token_found and not token_last:
                timer_end = now + 60
                print("Precision pickup → reset")

            if not buff_found:
                timer_end = 0

            token_last = token_found

            # -----------------
            # UI + TTS
            # -----------------

            if timer_end > now:
                remaining = timer_end - now

                if remaining > 20:
                    text = f"Precision: {int(remaining)}s"
                    label.config(text=text, fg="white")

                    tts_active = False

                else:
                    label.config(
                        text="Precision: Refresh",
                        fg="red"
                    )

                    if not tts_active:
                        tts_active = True
                        threading.Thread(
                            target=tts_loop,
                            daemon=True
                        ).start()

            else:
                label.config(
                    text="Precision: Nincs",
                    fg="gray"
                )
                tts_active = False

            time.sleep(0.1)

# -----------------------------
# THREAD INDÍTÁS
# -----------------------------

threading.Thread(
    target=detection_loop,
    daemon=True
).start()

root.mainloop()

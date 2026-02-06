import cv2
import numpy as np
import mss
import time
import pyttsx3
import threading
import tkinter as tk
from tkinter import font
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------
DEBUG = True
YOLO_MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5

TIMER_DURATION = 60
FRAME_CONFIRM = 3
PICKUP_COOLDOWN = 1.5

# -----------------------------
# GLOBALS
# -----------------------------
timer_end = 0
confirm_frames = 0
last_pickup_time = 0

# TTS
tts_active = False
tts_engine = None
tts_thread = None

model = YOLO(YOLO_MODEL_PATH)
print(f"YOLO model loaded: {YOLO_MODEL_PATH}")

# -----------------------------
# TTS loop
# -----------------------------
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
# TKINTER UI
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
# DETECTION LOOP
# -----------------------------
def detection_loop():
    global timer_end, confirm_frames, last_pickup_time, tts_active

    tts_thread = None

    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

        while True:
            now = time.time()

            raw = np.array(sct.grab(monitor))
            img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            results = model(img, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

            eye_detected = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # feltételezve, hogy class 0 az "eye"
                        eye_detected = True
                        break
                if eye_detected:
                    break

            # Frame megerősítés logika
            if eye_detected:
                confirm_frames += 1
            else:
                confirm_frames = 0

            # Ha elég frame megerősítette és cooldown lejárt → pickup
            if confirm_frames >= FRAME_CONFIRM and now - last_pickup_time > PICKUP_COOLDOWN:
                timer_end = now + TIMER_DURATION
                last_pickup_time = now
                tts_active = False
                print(f"Precision pickup! ({confirm_frames} frame megerősítve)")

            # -----------------
            # UI + TTS frissítés
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

            # -----------------
            # DEBUG PREVIEW
            # -----------------
            if DEBUG:
                preview = img.copy()

                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                preview,
                                f"Eye {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                # Info text
                cv2.putText(preview, f"confirm: {confirm_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                small_preview = cv2.resize(preview, (960, 540))
                cv2.imshow("YOLO Detection Preview", small_preview)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.08)

# -----------------------------
# Start
# -----------------------------
threading.Thread(
    target=detection_loop,
    daemon=True
).start()

root.mainloop()

# Cleanup
cv2.destroyAllWindows()
if tts_engine:
    tts_engine.stop()
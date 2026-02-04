import cv2
import numpy as np
import mss
import time
import pyttsx3

# -----------------------------
# TEMPLATE BETÖLTÉS
# -----------------------------

buff_templates = [
    cv2.imread("new1.png", 0),
    cv2.imread("new2.png", 0),
    cv2.imread("new3.png", 0),
    cv2.imread("new4.png", 0),
    cv2.imread("new5.png", 0),
    cv2.imread("new6.png", 0),
    cv2.imread("new7.png", 0),
    cv2.imread("new8.png", 0),
    cv2.imread("new9.png", 0),
    cv2.imread("new10.png", 0),
]

# Precision pickup particle template
particle_template = cv2.imread("precision_template.png", 0)

# -----------------------------
# BEÁLLÍTÁSOK
# -----------------------------

BUFF_THRESHOLD = 0.8
PARTICLE_THRESHOLD = 0.75

timer_end = 0
particle_last = False
alerted = False

# TTS
engine = pyttsx3.init()

# -----------------------------
# ROI-K
# -----------------------------

buff_roi = {
    "top": 40,
    "left": 0,
    "width": 1920,
    "height": 100
}

particle_roi = {
    "top": 300,
    "left": 800,
    "width": 300,
    "height": 300
}

# -----------------------------
# MSS LOOP
# -----------------------------

with mss.mss() as sct:

    while True:
        now = time.time()

        # Screenshotok
        buff_img = np.array(sct.grab(buff_roi))
        particle_img = np.array(sct.grab(particle_roi))

        buff_gray = cv2.cvtColor(buff_img, cv2.COLOR_BGR2GRAY)
        particle_gray = cv2.cvtColor(particle_img, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # BUFF DETECT
        # -------------------------

        buff_found = False

        for template in buff_templates:
            if template is None:
                continue

            res = cv2.matchTemplate(buff_gray, template, cv2.TM_CCOEFF_NORMED)

            if np.max(res) >= BUFF_THRESHOLD:
                buff_found = True
                break

        # -------------------------
        # PARTICLE DETECT
        # -------------------------

        particle_found = False

        if particle_template is not None:
            res2 = cv2.matchTemplate(
                particle_gray,
                particle_template,
                cv2.TM_CCOEFF_NORMED
            )

            if np.max(res2) >= PARTICLE_THRESHOLD:
                particle_found = True

        # -------------------------
        # TIMER LOGIKA
        # -------------------------

        # Pickup → reset
        if particle_found and not particle_last:
            timer_end = now + 60
            alerted = False
            print("Precision REFRESH → Timer reset 60s")

        # Buff eltűnt → stop
        if not buff_found:
            timer_end = 0
            alerted = False

        particle_last = particle_found

        # -------------------------
        # TIMER KIÍRÁS
        # -------------------------

        if timer_end > now:
            remaining = timer_end - now
            print(f"Hátralévő idő: {remaining:.1f}s")

            # 20s TTS alert
            if remaining <= 20 and not alerted:
                engine.say("Precision")
                engine.runAndWait()
                alerted = True

        else:
            print("Precision nincs / lejárt")

        # -------------------------
        # DEBUG ABLAKOK
        # -------------------------

        cv2.imshow("Buff ROI", buff_img)
        cv2.imshow("Particle ROI", particle_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)

import cv2
import numpy as np
import mss
import time
import pyttsx3

# -----------------------------
# BUFF TEMPLATEK (1x–10x)
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

# -----------------------------
# TOKEN / "PARTICLE" TEMPLATEK
# -----------------------------

token_templates = [
    cv2.imread("par1.png", 0),
    cv2.imread("par2.png", 0),
    cv2.imread("par3.png", 0),
    cv2.imread("par4.png", 0),
    cv2.imread("par5.png", 0),
    cv2.imread("precision_template.png", 0),
]

# -----------------------------
# BEÁLLÍTÁSOK
# -----------------------------

BUFF_THRESHOLD = 0.75
TOKEN_THRESHOLD = 0.7

timer_end = 0
token_last = False
alerted = False

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

token_roi = {
    "top": 300,
    "left": 800,
    "width": 300,
    "height": 300
}

# -----------------------------
# LOOP
# -----------------------------

with mss.mss() as sct:

    while True:
        now = time.time()

        buff_img = np.array(sct.grab(buff_roi))
        token_img = np.array(sct.grab(token_roi))

        buff_gray = cv2.cvtColor(buff_img, cv2.COLOR_BGR2GRAY)
        token_gray = cv2.cvtColor(token_img, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # BUFF DETECT
        # -------------------------

        buff_found = False

        for template in buff_templates:
            if template is None:
                continue

            h, w = template.shape

            res = cv2.matchTemplate(
                buff_gray,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            loc = np.where(res >= BUFF_THRESHOLD)

            for pt in zip(*loc[::-1]):
                buff_found = True
                cv2.rectangle(
                    buff_img,
                    pt,
                    (pt[0] + w, pt[1] + h),
                    (0, 255, 0),
                    2
                )

        # -------------------------
        # TOKEN DETECT (MULTI)
        # -------------------------

        token_found = False

        for template in token_templates:
            if template is None:
                continue

            h, w = template.shape

            res = cv2.matchTemplate(
                token_gray,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            loc = np.where(res >= TOKEN_THRESHOLD)

            for pt in zip(*loc[::-1]):
                token_found = True
                cv2.rectangle(
                    token_img,
                    pt,
                    (pt[0] + w, pt[1] + h),
                    (255, 0, 0),
                    2
                )

        # -------------------------
        # TIMER LOGIKA
        # -------------------------

        # Pickup → reset
        if token_found and not token_last:
            timer_end = now + 60
            alerted = False
            print("Precision pickup → Timer reset")

        # Buff eltűnt → stop
        if not buff_found:
            timer_end = 0
            alerted = False

        token_last = token_found

        # -------------------------
        # TIMER / ALERT
        # -------------------------

        if timer_end > now:
            remaining = timer_end - now
            print(f"Hátralévő idő: {remaining:.1f}s")

            if remaining <= 20 and not alerted:
                engine.say("Precision")
                engine.runAndWait()
                alerted = True
        else:
            print("Precision nincs / lejárt")

        # -------------------------
        # PREVIEW
        # -------------------------

        cv2.imshow("Buff ROI", buff_img)
        cv2.imshow("Token ROI", token_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)

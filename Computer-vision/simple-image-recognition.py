import depthai as dai
import cv2
import numpy as np

# === 1. Sett opp DepthAI pipeline ===
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)

# Økt oppløsning og FPS
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(60)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.video.link(xout.input)

# === 2. Oppdatert fargeområde (bedre rød/oransje-separasjon) ===
colors_hsv = {
    "red":    [((0, 130, 80), (8, 255, 255)), ((170, 130, 80), (179, 255, 255))],
    "orange": [((9, 120, 120), (20, 255, 255))],
    "green":  [((35, 30, 40), (85, 255, 255))],
    "blue":   [((85, 40, 40), (130, 255, 255))],
    "yellow": [((22, 100, 100), (32, 255, 255))]
}

color_bgr = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}

# === 3. Fargebalanse og gamma-korrigering ===
def correct_color_balance(frame_bgr, gamma=1.2, warm_shift=15):
    lookup = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    frame = cv2.LUT(frame_bgr, lookup)

    b, g, r = cv2.split(frame)
    r = cv2.add(r, warm_shift)
    b = cv2.subtract(b, warm_shift)
    frame = cv2.merge((b, g, r))

    return frame

# === 4. Kjør pipeline og prosesser videostrøm ===
with dai.Device(pipeline) as device:
    print("Koblet til OAK-D Lite...")
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        in_frame = video_queue.get()
        frame = in_frame.getCvFrame()

        frame = correct_color_balance(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, hsv_ranges in colors_hsv.items():
            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in hsv_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_total = cv2.bitwise_or(mask_total, mask)

            contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 800:
                    x, y, w, h = cv2.boundingRect(cnt)
                    color = color_bgr[color_name]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, color_name.upper(), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("OAK-D Color Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

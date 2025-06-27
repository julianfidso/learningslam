import depthai as dai
import cv2
import numpy as np
import time

# === Pipeline setup ===
pipeline = dai.Pipeline()

# --- Color camera ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(60)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# --- Mono cameras for stereo depth ---
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)

mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setSubpixel(True)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# --- Fargegrenser i HSV ---
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

# --- Fargekorrigering ---
def correct_color_balance(frame_bgr, gamma=1.2, warm_shift=15):
    lookup = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    frame = cv2.LUT(frame_bgr, lookup)
    b, g, r = cv2.split(frame)
    r = cv2.add(r, warm_shift)
    b = cv2.subtract(b, warm_shift)
    return cv2.merge((b, g, r))

# === Start device ===
with dai.Device(pipeline) as device:
    print("Koblet til OAK-D Lite")
    q_rgb = device.getOutputQueue("rgb", 4, False)
    q_depth = device.getOutputQueue("depth", 4, False)

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame().astype(np.uint16)

        display_w, display_h = 720, 405
        frame = cv2.resize(frame, (display_w, display_h))

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        frame = correct_color_balance(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_color = None
        for name, ranges in colors_hsv.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            if mask[cy, cx] > 0:
                detected_color = name
                break

        dh, dw = depth_frame.shape
        scaled_cx = int(cx * dw / w)
        scaled_cy = int(cy * dh / h)
        val = depth_frame[scaled_cy, scaled_cx]
        distance_m = val / 1000.0 if 300 <= val < 10000 else None

        # === Dybde statistikk ===
        valid_mask = (depth_frame >= 300) & (depth_frame < 10000)
        if np.any(valid_mask):
            depth_valid = depth_frame[valid_mask] / 1000.0
            dmin = np.min(depth_valid)
            dmax = np.max(depth_valid)
            dmean = np.mean(depth_valid)
            stats_text = f"Min: {dmin:.2f}m  Max: {dmax:.2f}m  Snitt: {dmean:.2f}m"
        else:
            stats_text = "Ingen gyldige dybdeverdier"

        depth_vis = ((np.clip(depth_frame, 300, 10000) - 300) / 9700 * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        depth_colored = cv2.resize(depth_colored, (display_w, display_h))

        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(depth_colored, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)

        label = ""
        color = (255, 255, 255)
        if detected_color:
            label += f"Farge: {detected_color.upper()}  "
            color = color_bgr[detected_color]
        if distance_m:
            label += f"Avstand: {distance_m:.2f} m"
        else:
            label += "Ingen gyldig dybde"

        frame_count += 1
        if time.time() - prev_time >= 1.0:
            fps = frame_count
            frame_count = 0
            prev_time = time.time()

        fps_label = f"FPS: {fps}"

        panel = np.zeros((70, display_w * 2, 3), dtype=np.uint8)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(panel, fps_label, (display_w * 2 - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        combined = np.hstack((frame, depth_colored))
        stacked = np.vstack((combined, panel))

        cv2.imshow("OAK-D RGB + Dybdevisning", stacked)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
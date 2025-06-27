import depthai as dai
import cv2
import numpy as np

# Setup pipeline
pipeline = dai.Pipeline()

# Mono-kamera
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

# Stereo depth
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setSubpixel(True)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)

config = stereo.initialConfig
config.setConfidenceThreshold(250)  # Mer robust måling
config.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
config.setLeftRightCheckThreshold(5)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Dybde-output
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Init device
with dai.Device(pipeline) as device:
    print("Koblet til OAK-D Lite")
    calib = device.readCalibration()
    print(f"Kalibrert baseline: {calib.getBaselineDistance():.2f} mm")

    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.5
    info_height = 40

    # Fargebar (TURBO)
    colorbar_base = np.linspace(255, 0, 256).astype(np.uint8)
    colorbar_base = np.tile(colorbar_base.reshape(-1, 1), (1, 40))
    colorbar_base = cv2.applyColorMap(colorbar_base, cv2.COLORMAP_TURBO)

    try:
        while True:
            in_depth = depth_queue.get()
            depth_frame = in_depth.getFrame().astype(np.uint16)

            h, w = depth_frame.shape
            cx, cy = w // 2, h // 2
            center_val = depth_frame[cy, cx]
            center_distance = center_val / 1000.0 if 300 <= center_val < 10000 else None

            # Fjern usikker dybde
            depth_frame[depth_frame < 300] = 10000
            depth_frame[depth_frame > 10000] = 10000

            # Normalisering og farge
            depth_vis = ((np.clip(depth_frame, 300, 10000) - 300) / 9700 * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

            # Støydemping (CPU)
            depth_color = cv2.bilateralFilter(depth_color, 9, 75, 75)

            # Marker midten
            cv2.drawMarker(depth_color, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)

            # Skaler visning
            depth_color_resized = cv2.resize(depth_color, (0, 0), fx=scale, fy=scale)
            colorbar = cv2.resize(colorbar_base, (colorbar_base.shape[1], depth_color_resized.shape[0]))

            # Meteretiketter (riktig vei nå: 10m øverst → 0.3m nederst)
            for i, m in zip([0, 64, 128, 192, 255], [10.0, 8.0, 6.0, 3.0, 0.3]):
                y = int(i * depth_color_resized.shape[0] / 256)
                cv2.putText(colorbar, f"{m:.1f}m", (2, y), font, 0.4, (255, 255, 255), 1)

            combined = np.hstack((depth_color_resized, colorbar))

            # Info stripe
            info_panel = np.zeros((info_height, combined.shape[1], 3), dtype=np.uint8)
            info_text = f"Senteravstand: {center_distance:.2f} m" if center_distance else "Ingen gyldig dybde i senter"
            cv2.putText(info_panel, info_text, (20, 27), font, 0.75, (255, 255, 255), 2)

            final_output = np.vstack((combined, info_panel))
            cv2.imshow("Dybdekart (OAK-D Lite)", final_output)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Manuelt avbrudd.")

    cv2.destroyAllWindows()

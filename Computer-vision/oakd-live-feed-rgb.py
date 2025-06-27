import depthai as dai
import cv2

# === 1. Lag pipeline ===
pipeline = dai.Pipeline()

# === 2. Legg til RGB-kamera node ===
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1280, 720)  # Bedre oppløsning for SLAM og visuell deteksjon
cam_rgb.setInterleaved(False)  # Viktig for OpenCV-kompatibilitet
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(30)  # Stabil oppdateringsfrekvens

# === 3. Opprett output node for å sende bildet til PC ===
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)  # Koble preview-strøm til output

# === 4. Koble til enheten og start pipeline ===
with dai.Device(pipeline) as device:
    print("Koblet til OAK-D Lite")

    # Hent output stream
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    try:
        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()  # Konverter til OpenCV-bilde

            # Vis bilde
            cv2.imshow("OAK-D RGB Preview", frame)

            # Avslutt med 'q'
            if cv2.waitKey(1) == ord('q'):
                print("Avslutter...")
                break

    except KeyboardInterrupt:
        print("Manuell avbrudd.")

    # Rydd opp
    cv2.destroyAllWindows()

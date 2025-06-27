import depthai as dai
import cv2
import numpy as np

# Opprett pipeline
pipeline = dai.Pipeline()

# Venstre mono-kamera
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_left.setFps(30)

# Hoyre mono-kamera
mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_right.setFps(30)

# Output streams
xout_left = pipeline.create(dai.node.XLinkOut)
xout_left.setStreamName("left")
mono_left.out.link(xout_left.input)

xout_right = pipeline.create(dai.node.XLinkOut)
xout_right.setStreamName("right")
mono_right.out.link(xout_right.input)

# Koble til enhet og kjor pipeline
with dai.Device(pipeline) as device:
    print("Koblet til OAK-D Lite (Stereo kameraer)")

    # Output-koer
    q_left = device.getOutputQueue("left", maxSize=4, blocking=False)
    q_right = device.getOutputQueue("right", maxSize=4, blocking=False)

    try:
        while True:
            frame_left = q_left.get().getCvFrame()
            frame_right = q_right.get().getCvFrame()

            # Kombiner bildene horisontalt
            combined_frame = np.hstack((frame_left, frame_right))

            # Vis kombinert bilde
            cv2.imshow("Stereo kameraer (venstre | hoyre)", combined_frame)

            if cv2.waitKey(1) == ord('q'):
                print("Avslutter...")
                break

    except KeyboardInterrupt:
        print("Manuelt avbrudd.")

    # Opprydding
    cv2.destroyAllWindows()
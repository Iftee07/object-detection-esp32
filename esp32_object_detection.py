import cv2
import numpy as np
import requests
import time
from ultralytics import YOLO

# ========================= CONFIG =========================
ESP32_IP = "10.131.215.1"
STREAM_URL = f"http://{ESP32_IP}:81/stream"

headers = {'User-Agent': 'Mozilla/5.0'}

print("🔄 Connecting...")

try:
    stream = requests.get(STREAM_URL, stream=True, headers=headers, timeout=10)
    stream.raise_for_status()
    print("✅ Connected! Starting detection...")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

model = YOLO("yolov8n.pt")   # Fastest model

bytes_data = b''
frame_count = 0
start_time = time.time()
yolo_every = 2               # Process YOLO every 2nd frame (increases FPS)

print("🎯 Press 'q' to quit")

while True:
    try:
        chunk = stream.raw.read(4096)   # Larger chunk = better
        if not chunk:
            break

        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                continue   # Skip bad frame (this fixes your error)

            # === YOLO only on every Nth frame for better FPS ===
            if frame_count % yolo_every == 0:
                results = model(frame, verbose=False, conf=0.4)
                annotated = results[0].plot(conf=True, labels=True)
            else:
                annotated = frame.copy()

            # FPS Counter
            frame_count += 1
            if frame_count % 15 == 0:
                fps = frame_count / (time.time() - start_time)
                cv2.putText(annotated, f"FPS: {fps:.1f} | YOLO:{yolo_every}th", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("ESP32-CAM YOLO Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Minor error (skipped): {e}")
        continue

stream.close()
cv2.destroyAllWindows()
print("Stopped.")
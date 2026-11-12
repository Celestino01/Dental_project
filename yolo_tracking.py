
import cv2
from collections import deque
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------

model = YOLO("yolo26m.pt")

allowed_classes = [
    i for i, name in model.names.items()
    if name.lower() in {"banana", "cup"}
]

BANANA_LENGTH_MM = 180.0  # Measure your banana
MM_PER_PIXEL = 0.4        # Placeholder: calibrate below
MAX_HISTORY = 50

banana_history = deque(maxlen=MAX_HISTORY)
tracked_banana_id = None
cup_rim_y = None

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot open camera")


# -----------------------------
# MOUSE CALIBRATION
# -----------------------------

def mouse_callback(event, x, y, flags, userdata):
    global cup_rim_y

    if event == cv2.EVENT_LBUTTONDOWN:
        cup_rim_y = y
        print(f"Cup rim set at Y={cup_rim_y}")


# -----------------------------
# DEPTH CALCULATION
# -----------------------------

def calculate_depth(top_y, rim_y):
    distance_above_rim_px = rim_y - top_y

    distance_above_rim_mm = (
        distance_above_rim_px * MM_PER_PIXEL
    )

    depth_mm = max(
        0.0,
        BANANA_LENGTH_MM - distance_above_rim_mm
    )

    depth_mm = min(depth_mm, BANANA_LENGTH_MM)

    percentage = (
        depth_mm / BANANA_LENGTH_MM
    ) * 100

    return depth_mm, percentage


# -----------------------------
# MAIN
# -----------------------------

def main():
    global tracked_banana_id

    cv2.namedWindow("Banana Depth Tracking")
    cv2.setMouseCallback(
        "Banana Depth Tracking",
        mouse_callback
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=allowed_classes,
                conf=0.20,
                imgsz=640,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            candidates = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = result.names[class_id].lower()

                    if label != "banana":
                        continue

                    if box.id is None:
                        continue

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    candidates.append({
                        "id": int(box.id[0]),
                        "box": (x1, y1, x2, y2),
                        "conf": float(box.conf[0])
                    })

            chosen = next(
                (
                    c for c in candidates
                    if c["id"] == tracked_banana_id
                ),
                None
            )

            if chosen is None and tracked_banana_id is None:
                if candidates:
                    chosen = max(
                        candidates,
                        key=lambda c: c["conf"]
                    )
                    tracked_banana_id = chosen["id"]

            if chosen is not None:
                x1, y1, x2, y2 = chosen["box"]

                top_x = (x1 + x2) // 2
                top_y = y1

                banana_history.append((top_x, top_y))

                cv2.circle(
                    annotated,
                    (top_x, top_y),
                    7,
                    (0, 0, 255),
                    -1
                )

                for i in range(1, len(banana_history)):
                    cv2.line(
                        annotated,
                        banana_history[i - 1],
                        banana_history[i],
                        (255, 0, 255),
                        2
                    )

                if cup_rim_y is not None:
                    depth_mm, percentage = calculate_depth(
                        top_y, cup_rim_y
                    )

                    cv2.putText(
                        annotated,
                        f"Depth: {depth_mm:.1f} mm",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"Inserted: {percentage:.1f}%",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

            else:
                cv2.putText(
                    annotated,
                    "Banana tracking lost",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            if cup_rim_y is not None:
                cv2.line(
                    annotated,
                    (0, cup_rim_y),
                    (annotated.shape[1], cup_rim_y),
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                annotated,
                "Click cup rim | R=Reset | Q=Quit",
                (20, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Banana Depth Tracking",
                annotated
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                tracked_banana_id = None
                banana_history.clear()

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

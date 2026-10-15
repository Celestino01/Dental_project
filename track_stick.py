import cv2
import numpy as np


# Starting color limits for bright green tape.
# We may need to adjust these for your lighting.
LOWER_GREEN = np.array([35, 80, 60], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

MIN_MARKER_AREA = 40  # Ignore tiny patches of green.


def find_marker(frame):
    """Return the marker's center and outline, or None."""

    # HSV separates color (hue) from saturation and brightness.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Matching pixels become white; everything else becomes black.
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Remove small isolated patches of noise.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find the boundaries of white regions.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_MARKER_AREA
    ]

    # Only accept one candidate to avoid choosing between green objects.
    if len(candidates) != 1:
        return None, mask

    contour = candidates[0]
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None, mask

    # Calculate the region's center of mass.
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]

    return (center_x, center_y, contour), mask


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open camera. Check macOS permissions.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a camera frame.")
                break

            marker, mask = find_marker(frame)

            if marker is None:
                message = "Tracking lost or ambiguous"
                color = (0, 0, 255)
            else:
                x, y, contour = marker
                center = (round(x), round(y))

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                message = f"Marker: x={x:.1f}, y={y:.1f} pixels"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                message,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.imshow("Stick tracking - press q to quit", frame)
            cv2.imshow("Green detection mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
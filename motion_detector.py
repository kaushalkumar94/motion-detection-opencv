import cv2
import time
import numpy as np


class MotionDetector:
    def __init__(self, source=0, min_contour_area=1000, threshold=50, display=True):
        self.video = cv2.VideoCapture(source)
        self.min_contour_area = min_contour_area
        self.threshold = threshold
        self.display = display
        self.first_frame = None
        self.motion_log = []

        # Adjust frame size for better performance
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray

    def detect_motion(self):
        while True:
            check, frame = self.video.read()
            if not check:
                break

            processed = self.process_frame(frame)

            if self.first_frame is None:
                self.first_frame = processed
                time.sleep(2)  # Allow camera to adjust
                continue

            delta_frame = cv2.absdiff(self.first_frame, processed)
            threshold_frame = cv2.threshold(delta_frame, self.threshold, 255, cv2.THRESH_BINARY)[1]
            threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

            contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue

                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Log motion event
                self.motion_log.append({
                    'timestamp': time.time(),
                    'location': (x, y, w, h),
                    'frame_size': frame.shape
                })

            if motion_detected:
                cv2.putText(frame, "MOTION DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.display:
                cv2.imshow("Motion Detection", frame)
                cv2.imshow("Threshold View", threshold_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('x'):
                    break
                elif key == ord('r'):  # Reset background
                    self.first_frame = self.process_frame(frame)
                elif key == ord('s'):  # Save snapshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"motion_{timestamp}.jpg", frame)

    def release(self):
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = MotionDetector()
    try:
        detector.detect_motion()
    finally:
        detector.release()
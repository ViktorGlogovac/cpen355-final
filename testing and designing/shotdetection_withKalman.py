from ultralytics import YOLO
import cv2
import math
import numpy as np
from detection import Detection  # Import from your existing files
from tracker import Tracker  # Import from your existing files
from nn_matching import NearestNeighborDistanceMetric  # Import from your existing files
from kalman_filter import KalmanFilter  # Import from your existing files
import datetime

class ShotDetector:
    def __init__(self):
        # Load the YOLO model with your custom weights
        self.model = YOLO("/Users/vikto/OneDrive/Documents/VESP/shot-tracker-ai/runs/detect/basketDetector7/weights/best.pt")
        self.class_names = ['ball', 'made', 'person', 'rim', 'shoot']

        # Load the video file
        self.cap = cv2.VideoCapture("TestVideos/1on1basket.mp4")

        # Initialize DeepSORT with Kalman Filter and Nearest Neighbor Matching
        max_cosine_distance = 0.2
        nn_budget = None

        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        # Variables to track ball and hoop positions
        self.previous_ball_position = None

        self.run()

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break  # End of the video or an error occurred

            # Perform object detection
            results = self.model(frame, stream=True)

            detections = []
            ball_bbox = None
            hoop_bbox = None

            # Loop through the detected results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    # Only process detections for "ball" and "rim"
                    if current_class == "rim" or current_class == "ball":
                        detection_bbox = np.array([x1, y1, x2, y2])
                        detections.append(Detection(detection_bbox, conf, cls))

                        # Mark the detected objects
                        color = (255, 0, 0) if current_class == "ball" else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, current_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Pass detections to the tracker (DeepSORT with Kalman Filter)
            self.tracker.predict()
            self.tracker.update(detections)

            # Iterate through tracked objects
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                # Get the bounding box and class id of the tracked object
                track_id = track.track_id
                class_id = track.class_id
                bbox = track.to_tlbr()  # Convert bounding box from top-left to bottom-right coordinates

                # Track basketball and hoop separately
                if class_id == 0:  # Basketball class
                    ball_bbox = bbox
                elif class_id == 1:  # Hoop class
                    hoop_bbox = bbox

                # Draw tracking boxes
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # ball (green), hoop (blue)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f"ID {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Implement score detection logic
            if ball_bbox is not None and hoop_bbox is not None:
                if self.previous_ball_position is not None and self.is_ball_scored(ball_bbox, hoop_bbox):
                    # Display the message on the frame
                    cv2.putText(frame, "Basket Scored!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    # Log the event to a text file
                    with open("basket_log.txt", "a") as log_file:
                        log_file.write(f"Basket scored at {datetime.datetime.now()}\n")
                    print("Basket Scored!")  # Print to console as well

                # Debugging output to check positions
                print(f"Ball Position: {ball_bbox}")
                print(f"Hoop Position: {hoop_bbox}")

                # Debugging IoU
                iou = self.calculate_iou(ball_bbox, hoop_bbox)
                print(f"IoU between ball and hoop: {iou}")

                self.previous_ball_position = ball_bbox

            # Resize the frame for display purposes
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
            cv2.imshow('Frame', frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def is_ball_scored(self, ball_bbox, hoop_bbox):
        """Check if the ball has been scored (passed through the hoop)."""
        # Check if the ball is entering the hoop (IoU overlap)
        if self.calculate_iou(ball_bbox, hoop_bbox) > 0.5:
            # Check if the ball is moving downward
            if ball_bbox[1] > self.previous_ball_position[1]:  # Y-coordinate is increasing
                return True
        return False

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        bbox1_area = w1 * h1
        bbox2_area = w2 * h2

        iou_value = inter_area / float(bbox1_area + bbox2_area - inter_area)
        return iou_value


if __name__ == "__main__":
    ShotDetector()

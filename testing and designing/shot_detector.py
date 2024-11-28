from ultralytics import YOLO
import cv2
import numpy as np

class ShotDetector:
    def __init__(self):
        self.model = YOLO("/Users/vikto/OneDrive/Documents/VESP/shot-tracker-ai/runs/detect/basketDetector-v6-1/weights/best.pt")
        self.class_names = ['made-basket']
        ##self.video_path = "TestVideos/free-throw.mp4"
        self.video_path = "TestVideos/hwss.mp4"
        self.cap = cv2.VideoCapture(self.video_path)

        # Initialize counters and states
        self.required_consecutive_frames = 3  # Reduced to handle lower frame rates
        self.cooldown_frames = 60  # Approx. 2 seconds at 30 FPS
        self.consecutive_frames = 0
        self.cooldown_counter = 0
        self.total_baskets = 0  # Counter for made baskets

        # Upscale factor for frames
        self.upscale_factor = 2

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Upscale the frame
            frame_upscaled = cv2.resize(frame, (0, 0), fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_LINEAR)

            # Optionally apply filters (e.g., sharpening)
            kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
            frame_upscaled = cv2.filter2D(frame_upscaled, -1, kernel)

            # Run detection on the upscaled frame
            results = self.model(frame_upscaled)

            basket_detected = False  # Reset for this frame

            # Process detections
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    class_name = self.class_names[cls]
                    if class_name == 'made-basket':
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Scale coordinates back to original frame size
                        x1 /= self.upscale_factor
                        y1 /= self.upscale_factor
                        x2 /= self.upscale_factor
                        y2 /= self.upscale_factor

                        # Draw bounding box on original frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # Put confidence score
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)
                        if conf > 0.5:  # Lowered confidence threshold
                            basket_detected = True

            # Handle cooldown and consecutive frames
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            if self.cooldown_counter == 0:
                if basket_detected:
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 0

                if self.consecutive_frames >= self.required_consecutive_frames:
                    self.total_baskets += 1
                    self.cooldown_counter = self.cooldown_frames
                    self.consecutive_frames = 0

            # Display the total number of made baskets
            cv2.putText(frame, f'Baskets Made: {self.total_baskets}', (frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Shot Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ShotDetector().run()

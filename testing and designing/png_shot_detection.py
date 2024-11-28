from ultralytics import YOLO
import cv2
import numpy as np

class ShotDetector:
    def __init__(self):
        self.model = YOLO("/Users/vikto/OneDrive/Documents/VESP/shot-tracker-ai/runs/detect/basketDetector-v6-1/weights/best.pt")
        self.class_names = self.model.names  # Use class names from the model
        self.image_path = "TestVideos/made-basket-ft.png"
        self.image = cv2.imread(self.image_path)

        # Upscale factor for the image
        self.upscale_factor = 2

    def run(self):
        # Check if the image was loaded properly
        if self.image is None:
            print(f"Failed to load image at {self.image_path}")
            return

        # Upscale the image
        image_upscaled = cv2.resize(self.image, (0, 0), fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_LINEAR)

        # Optionally apply filters (e.g., sharpening)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image_upscaled = cv2.filter2D(image_upscaled, -1, kernel)

        # Run detection on the upscaled image
        results = self.model(image_upscaled)

        # Process detections
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                class_name = self.class_names[cls]
                print(f"Detected class: {class_name}, Confidence: {conf}")

                if class_name == 'made-basket' and conf > 0.5:  # Adjust confidence threshold if needed
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Scale coordinates back to original image size
                    x1 /= self.upscale_factor
                    y1 /= self.upscale_factor
                    x2 /= self.upscale_factor
                    y2 /= self.upscale_factor

                    # Draw bounding box on the original image
                    cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put confidence score
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(self.image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        # Save or display the image with detections
        output_path = "output.png"
        cv2.imwrite(output_path, self.image)
        print(f"Output saved to {output_path}")

        # Optionally display the image
        cv2.imshow('Detection', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ShotDetector().run()

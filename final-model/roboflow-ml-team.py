import cv2
import requests
import numpy as np

# Roboflow configuration
ROBOFLOW_API_KEY = "upR0TpbLVIlvsGJwoHe5"   # Replace with your Roboflow API key
ROBOFLOW_MODEL = "score-keep-ml"            # Replace with your Roboflow model name
ROBOFLOW_VERSION = "1"                      # Replace with your model's version number
ROBOFLOW_SIZE = 416                         # Replace with your model's input size if different

# Set base confidence threshold (float between 0 and 1)
CONFIDENCE_THRESHOLD = 0.3

ROBOFLOW_URL = (
    f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}"
    f"?api_key={ROBOFLOW_API_KEY}&format=json&stroke=5&labels=true&confidence={CONFIDENCE_THRESHOLD}"
)

# Open the video file or capture device
video_path = 'saints-bucket.mov'  # Replace with your video path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(
    'saints-bucket-annotated.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Resize the frame to the model's input size
    resized_frame = cv2.resize(frame, (ROBOFLOW_SIZE, ROBOFLOW_SIZE))

    # Encode frame as JPEG
    retval, buffer = cv2.imencode('.jpg', resized_frame)
    img_bytes = buffer.tobytes()

    # Prepare payload for POST request
    files = {
        'file': ('frame.jpg', img_bytes, 'image/jpeg')
    }

    # Send the image to the Roboflow API
    response = requests.post(ROBOFLOW_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        detections = result.get('predictions', [])

        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']

            # Apply confidence thresholds
            if class_name.lower() == 'made-basket':
                if confidence < 0.5:
                    continue  # Skip detections of 'made-basket' below 0.5 confidence
            else:
                if confidence < 0.3:
                    continue  # Skip detections of other classes below 0.3 confidence

            # Extract bounding box coordinates
            x = detection['x']
            y = detection['y']
            w = detection['width']
            h = detection['height']

            # Scale coordinates back to original frame size
            x_scale = frame_width / ROBOFLOW_SIZE
            y_scale = frame_height / ROBOFLOW_SIZE

            x = x * x_scale
            y = y * y_scale
            w = w * x_scale
            h = h * y_scale

            # Convert center coordinates to top-left and bottom-right
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width - 1, x2)
            y2 = min(frame_height - 1, y2)

            # Assign colors based on class
            if class_name.lower() == 'made-basket':
                color = (0, 255, 0)  # Green for made-basket
            elif class_name.lower() == 'shoot':
                color = (255, 0, 230)  # Custom color for 'shoot'
            else:
                color = (255, 64, 64)  # Red for default

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label = f"{class_name}: {confidence * 100:.1f}%"

            # If the detection is a person, apply team detection
            if class_name.lower() == 'person':
                # Extract the player's bounding box from the original frame
                player_roi = frame[y1:y2, x1:x2]

                # Convert to grayscale
                gray_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2GRAY)

                # Apply threshold to isolate dark pixels (intensity less than 50)
                _, dark_mask = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)

                # Count dark pixels and total pixels
                num_dark_pixels = cv2.countNonZero(dark_mask)
                total_pixels = gray_roi.size
                dark_ratio = num_dark_pixels / total_pixels

                # Decide on the team based on the dark_ratio
                if dark_ratio > 0.3:
                    team_label = 'Dark Team'
                else:
                    team_label = 'Light Team'

                label += f" - {team_label}"

            # Put label above bounding box
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    else:
        print("Error:", response.status_code, response.text)
        # Optionally, skip drawing on this frame or handle the error

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

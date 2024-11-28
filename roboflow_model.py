import cv2
from roboflow import Roboflow

# Initialize Roboflow model
rf = Roboflow(api_key="upR0TpbLVIlvsGJwoHe5")
project = rf.workspace("cpen-355").project("score-keep-ml")
model = project.version(1).model

# Open video file
cap = cv2.VideoCapture('TestVideos/stg-fox.mp4')

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if mp4v doesn't work
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Predict on the current frame (no need to convert to bytes)
        prediction = model.predict(frame, confidence=40, overlap=30).json()

        # Draw bounding boxes and confidence levels
        for obj in prediction['predictions']:
            # Bounding box coordinates
            x1, y1 = int(obj['x'] - obj['width'] / 2), int(obj['y'] - obj['height'] / 2)
            x2, y2 = int(obj['x'] + obj['width'] / 2), int(obj['y'] + obj['height'] / 2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put confidence and label
            label = f"{obj['class']} {obj['confidence']:.2f}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with the drawings to the output video
        out.write(frame)

        # Display the resulting frame (optional)
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

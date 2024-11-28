import requests
import cv2
import base64

# Replace with your actual Roboflow API key and model endpoint
api_key = "upR0TpbLVIlvsGJwoHe5"
model_endpoint = "https://api.roboflow.com/score-keep-ml/version/1/predict"

# Initialize video capture
cap = cv2.VideoCapture("saints-vc.mov")

def get_predictions(frame):
    _, encoded_image = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(encoded_image).decode('utf-8')

    response = requests.post(model_endpoint,
                             headers={'Authorization': f'Bearer {api_key}'},
                             json={'image': base64_image})
    if response.status_code != 200:
        print("Failed to get valid response from the server:", response.text)
        return None
    return response.json()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    predictions = get_predictions(frame)
    if predictions is None:
        continue  # Skip this frame if predictions are None

    # Check if 'predictions' key exists in the response
    if 'predictions' in predictions:
        for prediction in predictions['predictions']:
            x1 = int(prediction['x'])
            y1 = int(prediction['y'])
            x2 = int(prediction['width']) + x1
            y2 = int(prediction['height']) + y1
            class_name = prediction['class']
            confidence = prediction['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        print("No 'predictions' key in response:", predictions)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

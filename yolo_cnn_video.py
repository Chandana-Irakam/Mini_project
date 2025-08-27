from ultralytics import YOLO
import cv2
import tensorflow as tf

# Load YOLOv8
yolo_model = YOLO("yolov8n.pt")  # will download automatically if not present

# Load CNN
cnn_model = tf.keras.models.load_model("violence_classifier.h5")

# Video path
video_path = r"C:\Users\sures\Downloads\V_126.mp4"
cap = cv2.VideoCapture(video_path)

predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = yolo_model(frame)
    annotated_frame = results[0].plot()

    # CNN Prediction
    frame_resized = cv2.resize(frame, (128,128))
    frame_norm = frame_resized / 255.0
    frame_input = frame_norm.reshape(1,128,128,3)
    pred = cnn_model.predict(frame_input, verbose=0)
    predictions.append(pred[0][0])

    cv2.imshow("Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

average_pred = sum(predictions) / len(predictions)
final_label = "Violence" if average_pred > 0.5 else "Non-Violence"
print("Final Video Prediction:", final_label)

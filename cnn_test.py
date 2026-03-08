import cv2
import tensorflow as tf
import numpy as np

# Load your trained Stampede Detection CNN model
cnn_model = tf.keras.models.load_model("stampede_model_best.h5")  # update with your filename

# Path to video
video_path = r"C:\Users\sures\Downloads\3331913-hd_1920_1080_30fps.mp4"
# Open video
cap = cv2.VideoCapture(video_path)

predictions = []
frame_count = 0

print("Analyzing video... Please wait.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize and normalize frame
    frame_resized = cv2.resize(frame, (128, 128))
    frame_norm = frame_resized / 255.0
    frame_input = np.expand_dims(frame_norm, axis=0)

    # Predict (no display of result)
    pred = cnn_model.predict(frame_input, verbose=0)
    predictions.append(pred[0][0])

    # Show the video frame (just playback)
    cv2.imshow("Stampede Detection (Analyzing...)", frame)

    # Press 'q' to quit early if needed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Compute final prediction after processing all frames
average_pred = np.mean(predictions)
final_label = "Stampede Alert 🚨" if average_pred > 0.4 else "Normal Crowd ✅"

print("===================================")
print(f"Total Frames Analyzed: {frame_count}")
print(f"Average Prediction Score: {average_pred:.4f}")
print(f"Final Video Classification: {final_label}")
print("===================================")

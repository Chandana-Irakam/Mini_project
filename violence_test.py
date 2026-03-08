import cv2
import tensorflow as tf
import numpy as np
import sys
import json
import os

def analyze_video(video_path):
    """Analyze video for violence detection and return structured results"""
    try:
        # Load the trained CNN model
        cnn_model = tf.keras.models.load_model("violence_classifier_v2_best.h5")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "error": f"Could not open video file: {video_path}",
                "success": False
            }
        
        predictions = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize and normalize frame
            frame_resized = cv2.resize(frame, (128, 128))
            frame_norm = frame_resized / 255.0
            frame_input = np.expand_dims(frame_norm, axis=0)
            
            # Predict
            pred = cnn_model.predict(frame_input, verbose=0)
            predictions.append(pred[0][0])
        
        cap.release()
        
        # Compute final prediction
        if len(predictions) == 0:
            return {
                "error": "No frames could be processed from the video",
                "success": False
            }
        
        average_pred = np.mean(predictions)
        final_label = "Violence" if average_pred > 0.5 else "Non-Violence"
        
        return {
            "success": True,
            "total_frames": frame_count,
            "average_confidence": float(average_pred),
            "final_classification": final_label,
            "model_type": "violence"
        }
        
    except Exception as e:
        return {
            "error": f"Error during analysis: {str(e)}",
            "success": False
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python violence_test_modified.py <video_path>", "success": False}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = analyze_video(video_path)
    print(json.dumps(result))

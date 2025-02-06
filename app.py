from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
MODEL_PATH = 'models/final_model6.h5'

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load model
model = keras.models.load_model(MODEL_PATH)

# Directories for uploaded videos and extracted frames
UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Function to square crop the frames
def square_crop_frame(image):
    """Crop the given image to a square format."""
    height, width = image.shape[:2]
    min_dimension = min(height, width)
    start_x = (width - min_dimension) // 2
    start_y = (height - min_dimension) // 2
    return image[start_y:start_y + min_dimension, start_x:start_x + min_dimension]

# Function to process video frames
def process_video_frames(video_path, max_frames=0, resize_dims=(IMG_SIZE, IMG_SIZE)):
    """Extract and process frames from a video file."""
    capture = cv2.VideoCapture(video_path)
    processed_frames = []
    try:
        while True:
            read_success, frame = capture.read()
            if not read_success:
                break
            frame = square_crop_frame(frame)
            frame = cv2.resize(frame, resize_dims)
            frame = frame[..., ::-1]  # Convert BGR to RGB
            processed_frames.append(frame)

            if max_frames > 0 and len(processed_frames) >= max_frames:
                break
    finally:
        capture.release()
    return np.array(processed_frames)

# Function to extract features from video frames using the feature extractor
def extract_video_features(video_frames, feature_extractor):
    """Extract features from video frames using a feature extractor."""
    features = []
    for frame in video_frames:
        frame = np.expand_dims(frame, axis=0)
        feature = feature_extractor.predict(frame)
        features.append(feature)
    
    features = np.array(features)
    return features.squeeze()

# Function to build feature extractor model
def build_feature_extractor(model_name='ResNet50'):
    """Build the feature extractor model."""
    base_model_class = getattr(keras.applications, model_name)
    base_model = base_model_class(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = getattr(keras.applications, model_name.lower()).preprocess_input

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)
    outputs = base_model(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_name}_feature_extractor")
    return model

# Feature extractor instance
feature_extractor = build_feature_extractor('ResNet50')

# Function to prepare input for the final model prediction
def prepare_input(features, max_seq_length=MAX_SEQ_LENGTH):
    """Prepare features with masks for model prediction."""
    num_features = features.shape[-1]
    input_features = np.zeros((1, max_seq_length, num_features), dtype="float32")
    input_mask = np.zeros((1, max_seq_length), dtype=bool)

    num_frames = features.shape[0]
    frames_to_use = min(max_seq_length, num_frames)
    input_features[0, :frames_to_use] = features[:frames_to_use]
    input_mask[0, :frames_to_use] = True

    return input_features, input_mask

@app.route("/upload-video", methods=["POST"])
def upload_video():
    """Handle video file upload, frame extraction, and deepfake detection."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Secure and save video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)

    # Process video and get the frames
    video_frames = process_video_frames(video_path, max_frames=MAX_SEQ_LENGTH)
    video_features = extract_video_features(video_frames, feature_extractor)
    input_features, input_mask = prepare_input(video_features)

    # Frame-wise predictions
    frame_predictions = []
    frame_urls = []
    for i in range(video_features.shape[0]):
        # Prepare a sequence with one frame and pad it to max_seq_length
        temp_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        temp_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)
        temp_features[0, 0] = video_features[i]
        temp_mask[0, 0] = True

        # Predict for the current frame
        frame_prediction = model.predict([temp_features, temp_mask])

        # Threshold-based frame prediction (same as in your first code)
        frame_predictions.append("REAL" if frame_prediction[0] > 0.59136045 else "FAKE")

        # Save the frame image to the FRAME_FOLDER
        frame_filename = os.path.join(FRAME_FOLDER, f"frame_{i}.jpg")
        cv2.imwrite(frame_filename, video_frames[i])  # Save frame as an image

        # Add the URL of the frame to the frame_urls list
        frame_urls.append(f"http://localhost:3000/frames/frame_{i}.jpg")

    # Final prediction logic based on frame predictions
    fake_count = frame_predictions.count("FAKE")
    final_prediction = "FAKE" if fake_count >= 3 else "REAL"

    return jsonify({
        "frame_predictions": frame_predictions,
        "frame_urls": frame_urls,
        "final_prediction": final_prediction
    })

@app.route("/frames/<filename>")
def serve_frame(filename):
    """Serve the extracted frame images."""
    return send_from_directory(FRAME_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
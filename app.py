from flask import Flask, request, jsonify
import os
import urllib.request
import cv2
import numpy as np
import insightface

# initialize Flask app
app = Flask(__name__)

# function to download model to /tmp if missing
def download_model():
    model_url = "https://drive.google.com/uc?export=download&id=1ow4J2gNhbIG3Scrqfm_I2O31OR_kQ1M0"
    dest_path = "/tmp/inswapper_128.onnx"
    if not os.path.exists(dest_path):
        print("Downloading inswapper_128.onnx ...")
        urllib.request.urlretrieve(model_url, dest_path)
        print("Model downloaded!")
    return dest_path

# download model at startup
model_path = download_model()

# initialize face analysis
app_insight = insightface.app.FaceAnalysis(name="buffalo_l")
app_insight.prepare(ctx_id=0, det_size=(640, 640))

# load swapper from downloaded ONNX
swapper = insightface.model_zoo.get_model(model_path)
swapper.prepare(ctx_id=0)

# define Flask route
@app.route("/swap", methods=["POST"])
def swap_faces():
    try:
        # get uploaded files
        source_file = request.files.get("source")
        target_file = request.files.get("target")
        if not source_file or not target_file:
            return jsonify({"error": "source and target images are required"}), 400
        
        # convert to numpy arrays
        source_arr = np.frombuffer(source_file.read(), np.uint8)
        target_arr = np.frombuffer(target_file.read(), np.uint8)
        source_img = cv2.imdecode(source_arr, cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(target_arr, cv2.IMREAD_COLOR)
        
        # detect faces
        source_faces = app_insight.get(source_img)
        target_faces = app_insight.get(target_img)
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            return jsonify({"error": "no face detected in one or both images"}), 400
        
        source_face = source_faces[0]
        
        # swap each face in target
        result = target_img.copy()
        for face in target_faces:
            result = swapper.get(result, face, source_face, paste_back=True)
        
        # encode back to jpg
        _, buffer = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return buffer.tobytes(), 200, {"Content-Type": "image/jpeg"}
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# test hello route
@app.route("/")
def hello():
    return "Face Swap API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

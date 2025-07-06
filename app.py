from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from io import BytesIO

# Initialize Flask
app = Flask(__name__)

# initialize insightface
app_insight = FaceAnalysis(name="buffalo_l")
app_insight.prepare(ctx_id=0, det_size=(640,640))  # same as notebook

# load the same swapper
swapper = insightface.model_zoo.get_model('inswapper_128', download=True)

@app.route('/swap', methods=['POST'])
def swap_faces():
    # verify POSTed files
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({"error": "Provide both source and target images."}), 400
    
    source_file = request.files['source']
    target_file = request.files['target']
    
    # read bytes into numpy
    source_bytes = np.frombuffer(source_file.read(), np.uint8)
    target_bytes = np.frombuffer(target_file.read(), np.uint8)
    
    source_img = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
    target_img = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
    
    # detect faces in both images
    source_faces = app_insight.get(source_img)
    target_faces = app_insight.get(target_img)
    
    if len(source_faces) == 0 or len(target_faces) == 0:
        return jsonify({"error": "No face detected in one of the images"}), 400
    
    # get first face in source
    new_face = source_faces[0]
    
    # same logic as notebook
    res = target_img.copy()
    for face in target_faces:
        res = swapper.get(res, face, new_face, paste_back=True)
    
    # sharpen
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    res = cv2.filter2D(res, -1, sharpen_kernel)
    
    # encode to JPEG to return
    _, img_encoded = cv2.imencode('.jpg', res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return send_file(BytesIO(img_encoded.tobytes()), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

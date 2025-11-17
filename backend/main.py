import os
from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pdf2image import convert_from_bytes
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf_to_video(pdf_bytes, output_path):
    # Convert PDF pages to images
    images = convert_from_bytes(pdf_bytes, fmt="jpeg")
    if not images:
        raise IOError("No images extracted from PDF")
    
    # Set frame size and video writer
    frame = np.array(images[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1  # 1 frame per second (can be changed)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        frame = np.array(img)
        # YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    out.release()

@app.route('/process_pdf/', methods=['POST'])
def upload_and_process_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF uploaded'}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_filepath)

    output_filename = 'output_' + filename.rsplit('.', 1)[0] + '.mp4'
    output_filepath = os.path.join(app.config['RESULT_FOLDER'], output_filename)

    try:
        with open(input_filepath, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
            process_pdf_to_video(pdf_bytes, output_filepath)
    except Exception as e:
        return jsonify({'error': 'Processing failed', 'message': str(e)}), 500

    return jsonify({'output_video': f'/download_output?path={output_filename}'})

@app.route('/download_output')
def download_output():
    path = request.args.get('path')
    if not path:
        abort(400, 'Missing path parameter')
    if '..' in path or path.startswith('/'):
        abort(400, 'Invalid path parameter')
    full_path = os.path.join(app.config['RESULT_FOLDER'], path)
    if not os.path.exists(full_path):
        abort(404, 'File not found')
    return send_from_directory(app.config['RESULT_FOLDER'], path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

import os
from flask import Flask, request, jsonify
from detector import Detector

FILENAME = 'frame.png'
UPLOAD_FOLDER = 'upload/'
ALLOWED_FILETYPES = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
detector = Detector();

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_FILETYPES:
        return jsonify({'error': 'Invalid file type'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], FILENAME)
    file.save(file_path)
    det = detector.predict(file_path)

    return det

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

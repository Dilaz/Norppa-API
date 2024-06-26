import sys
from dotenv import load_dotenv
from flask import jsonify
from edgetpumodel import EdgeTPUModel
import logging

MODEL='model/best-transfer-int8.tflite'
MINIMUM_CONFIDENCE = 0.5

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.initModel()

    def initModel(self):
        self.model = EdgeTPUModel(MODEL, 'names.yaml', conf_thresh=MINIMUM_CONFIDENCE, iou_thresh=0.45, v8=False)
        self.input_size = self.model.get_image_size()
        logger.info('Model image size: %d' % self.input_size[0])
    
    def predict(self, frame):
        _, outputs = self.model.predict(frame)
        return jsonify(outputs[::-1])

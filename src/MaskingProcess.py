from torch.multiprocessing import set_start_method

from torch.multiprocessing import Lock
import cv2
import numpy as np
import mediapipe as mp

segmenter = None
set_start_method('spawn', force=True)

def load_mask_processor():
    global segmenter
    if segmenter is None:
        segmenter = MediapipeSegmenter()
    return segmenter

def worker_init_fn(worker_id):
    global segmenter
    if worker_id == 0:  # Only the first worker loads the model
        segmenter = load_mask_processor()

class MediapipeSegmenter:
    _instance = None  # Singleton instance
    _lock = Lock()
    _process_lock = Lock()

    # Attempt at only one instance at a time
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MediapipeSegmenter, cls).__new__(cls)
                    cls._instance.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        return cls._instance

    def process(self, image):
        with self._process_lock:

            # Read the image
            height, width, _ = image.shape
            image = (image * 255).astype(np.uint8)

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform segmentation
            result = self._instance.segmenter.process(image_rgb)

            # Extract the segmentation mask between background vs person
            mask = result.segmentation_mask

            # Threshold the mask to create a binary mask for the person
            _, binary_mask = cv2.threshold(mask, 0.25, 1, cv2.THRESH_BINARY)
        return binary_mask

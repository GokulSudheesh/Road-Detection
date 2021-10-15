import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Lane:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.model = load_model("./models/my_model01.h5")

    def road_lanes(self, *, image, image_layer):
        small_image = cv2.resize(image, (128, 128))
        small_image = np.array(small_image)
        small_image = small_image[None, :, :, :]

        # Outputs a BnW image (1 channel) (0-1) * 255
        prediction = np.argmax(self.model.predict(small_image)[0], axis=-1) * 255

        # Converting to three channel (RGB). Stack arrays [0..] [0...] [prediction]
        blanks = np.zeros_like(prediction).astype(np.uint8)
        lane_drawn = np.dstack((blanks, blanks, prediction))

        lane_img = cv2.resize(lane_drawn.astype(np.float32), (image.shape[1], image.shape[0]))
        result = cv2.addWeighted(image_layer, 0.004, lane_img, 0.004, 0, dtype=cv2.CV_32F)
        return result

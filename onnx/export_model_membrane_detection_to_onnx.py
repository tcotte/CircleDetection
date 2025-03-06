import typing

import cv2
import numpy as np
import onnxruntime
from matplotlib import pyplot as plt


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        params image: image array of size (C, H, W) to be normalized
        returns: normalized image
        """
        norm_array = []
        for array_channel, m, s in zip(image, self.mean, self.std):
            norm_array.append((array_channel - m) / s)
        return np.asarray(norm_array)


def get_crop_values(img_shape: typing.Tuple, marge: float, coords: [int, int, int, int]) -> [int, int, int, int]:
    """
    Get top-left-corner and right-bottom-corner of bounding box surrounding the membrane
    :param img_shape: Original picture shape
    :param marge: marge percentage. This value should be in interval [0; 1]
    :param coords: detected membrane coordinates
    """
    assert (marge > 0) and (marge < 1), "Marge percentage can't be negative or superior to 1"

    x0, y0, x1, y1 = coords
    width = (x1 - x0) * img_shape[0]
    height = (y1 - y0) * img_shape[1]

    y0, y1 = round(y0 * img_shape[0] - marge * height), round(y1 * img_shape[0] + marge * height)
    x0, x1 = round(x0 * img_shape[1] - marge * width), round(x1 * img_shape[1] + marge * width)

    return x0, y0, x1, y1


if __name__ == '__main__':
    onnx_modle_path: typing.Final[str] = '../models/Legionellas/membrane_detector.onnx'
    inference_size: typing.Final[int] = 1024
    image_path: typing.Final[str] = '<image_path>'

    image = cv2.imread(image_path)
    r = 1556
    crop_image = image

    input = cv2.resize(crop_image, (inference_size, inference_size))

    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    input = input.transpose((2, 0, 1)) / 255.

    # normalize picture
    input = normalize(input)

    # expand dims to get np array shape like batch_size, channels, height, width
    input = np.expand_dims(input.astype(np.float32), axis=0)

    ort_session = onnxruntime.InferenceSession(onnx_modle_path, providers=['CPUExecutionProvider'])

    input_name = ort_session.get_inputs()[0].name

    onnxruntime_input = {ort_session.get_inputs()[0]: input}

    # onnxruntime returns a list of outputs
    predictions = ort_session.run(['bboxes', 'class_logits', 'x_prob'], onnxruntime_input)

    membrane_presence = float(np.squeeze(predictions[-1].flatten(), axis=0)) > 0.5
    if membrane_presence:
        normalized_coords = predictions[0].flatten().tolist()
        x0, y0, x1, y1 = get_crop_values(crop_image.shape[:2], 0.2, normalized_coords)

        plt.title(
            f"Membrane found with {round(float(np.squeeze(predictions[-1].flatten(), axis=0)) * 100)}% of confidence")
        cv2.rectangle(crop_image, (x0, y0), (x1, y1), (0, 0, 255), 5)

    else:
        plt.title(
            f"Membrane not found with {round(float(np.squeeze(predictions[-1].flatten(), axis=0)) * 100)}% of confidence")

    plt.imshow(crop_image[:, :, ::-1])
    plt.show()

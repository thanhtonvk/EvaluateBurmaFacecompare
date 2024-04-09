from PIL import Image as PilImage
import numpy as np
from onnxruntimeInfer import inferenceSession
import cv2


class FaceParts:
    EYE = "EYE"
    NOSE = "NOSE"
    MOUTH = "MOUTH"


FACE_PARTS_MAP = {
    0: FaceParts.EYE,
    1: FaceParts.NOSE,
    2: FaceParts.MOUTH,
}


def is_full_face(labels):
    if (
            FaceParts.NOSE in labels
            and FaceParts.MOUTH in labels
            # and FaceParts.EYE in labels
            # and labels.count(FaceParts.EYE) >= 2
    ):
        return True
    return False


def load_model(model_path):
    model = inferenceSession(model_path)
    return model


class FacePartDetection:
    def __init__(self, model_path='best_face_part.onnx', imgsz=(128, 128)):
        self.model_path = model_path
        self.model = load_model(model_path)
        self.imgsz = imgsz

    def detect(self, pil_image: PilImage.Image):
        np_image = np.array(pil_image)
        # get shape of raw image
        img_width, img_height, _ = np_image.shape
        ratio_w = self.imgsz[0] / img_width
        ratio_h = self.imgsz[1] / img_height

        np_image = cv2.resize(np_image, self.imgsz).astype(np.float32) / 255.0
        # convert to ONNX model format
        np_image = np_image.transpose((2, 0, 1))
        np_image = np.expand_dims(np_image, 0)
        ort_inputs = {self.model.get_inputs()[0].name: np_image}
        result = self.model.run(None, ort_inputs)[0]
        if len(result) > 0:
            boxes = result[:, 1:5]
            scores = result[:, 6:].flatten()
            classes = result[:, 5:6].astype(int).flatten()
            labels = [FACE_PARTS_MAP[i] for i in classes]
            # resize bbox to raw size
            boxes[:, 0:1] /= ratio_h
            boxes[:, 1:2] /= ratio_w
            boxes[:, 2:3] /= ratio_h
            boxes[:, 3:4] /= ratio_w
            return is_full_face(labels), labels, scores, boxes
        return False, [], [], []


if __name__ == "__main__":
    face_part = FacePartDetection()
    import os

    image = PilImage.open(
        "/Users/tonsociu/Downloads/f368006f0a7ddc23856c.jpg").convert("RGB")
    result = face_part.detect(image)
    print(result)

import torch
from torchvision import transforms
from PIL import Image as PilImage
from onnxruntimeInfer import inferenceSession

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(eval_model):
    net = inferenceSession(eval_model)
    return net


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class FaceQuality:
    SIZE = (112, 112)
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(self, eval_model='face_quality.onnx'):
        self.model = load_model(eval_model)

    def preprocessing(self, face: PilImage.Image):
        face = self.transform(face).to(device)
        face = face.unsqueeze(0)
        face = to_numpy(face)
        return face

    def predict(self, face: PilImage.Image):
        face = self.preprocessing(face)
        ort_inputs = {self.model.get_inputs()[0].name: face}
        ort_outs = self.model.run(None, ort_inputs)
        score = ort_outs[0][0][0]
        return score

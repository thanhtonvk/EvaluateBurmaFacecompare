import numpy as np
import argparse, os
from torchvision import transforms
from PIL import Image
import torch
from model import Model
from utils import renorm


class FaceRecognitionModel():
  def __init__(self) -> None:
    self.opt = argparse.Namespace()
    self.opt.num_features = 512
    self.opt.pretrained = ""
    self.opt.save_path = 'ver_2.2.1_model.pth'
    self.opt.imgH = 112
    self.opt.imgW = 112
    self.device = 'cpu'

    self.model = Model(self.opt)
    self.model = torch.nn.DataParallel(self.model)
    if os.path.exists(self.opt.save_path):
      if self.device != "cpu":
          print("cuda")
          map_location = lambda storage, loc: storage.cuda()
      else:
          map_location = 'cpu'
    ckpt = torch.load(self.opt.save_path, map_location=map_location)
    self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # self.opt.logging.info('Trained model is loaded')

    self.model.eval()
    self.transformer = transforms.Compose([
        transforms.Resize((self.opt.imgH, self.opt.imgW)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    self.metric = torch.nn.CosineSimilarity(dim=-1)

  def compare(self, source_img, target_img):
    source_img = Image.fromarray(renorm(np.array(source_img)))
    target_img = Image.fromarray(renorm(np.array(target_img)))

    tensor_1 = self.transformer(source_img)
    tensor_2 = self.transformer(target_img)
    inputs = torch.stack([tensor_1, tensor_2], dim=0)
    if self.device != "cpu":
        inputs = inputs.cuda()
    with torch.no_grad():
      embed = self.model(inputs)

    scores = self.metric(embed[0], embed[1])

    return self.rescale_score(scores)

  def rescale_score(self, scores):
    # if not CPU:
    #   scores = scores.detach().cpu()
    # print(scores)
    scores = scores.detach().cpu().numpy()
    scores = np.where(scores >= 0.35, 0.35, scores)
    scores = np.where(scores <= 0.02, 0.02, scores)
    scores = (scores - 0.02) / 0.33

    return scores


  def compare_embedding(self, source_embedding, target_embedding_list):
    embedding_2 = torch.stack(target_embedding_list, dim=0)
    scores = self.metric(source_embedding.unsqueeze(dim=0), embedding_2)

    return self.rescale_score(scores).tolist()

  def get_face_embedding(self, face_pil):
    source_img = Image.fromarray(renorm(np.array(face_pil)))

    tensor_1 = self.transformer(source_img)
    inputs = torch.stack([tensor_1], dim=0)
    if self.device != "cpu":
        inputs = inputs.cuda()
    with torch.no_grad():
      embed = self.model(inputs)
    return embed[0]

  def get_face_embedding_size(self):
    return 512



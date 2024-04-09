import torch, os
from iresnet import iresnet50



class Model(torch.nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.backbone = iresnet50()
        self._init_weights()

    def _init_weights(self):
        if os.path.isfile(self.opt.pretrained):
            self.backbone.load_state_dict(
                torch.load(self.opt.pretrained, map_location='cpu')
            )
            self.opt.logging.info('Pre-trained model is used')
    
    def forward(self, x):
        feat = self.backbone(x)
        return feat
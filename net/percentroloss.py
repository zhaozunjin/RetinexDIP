from collections import namedtuple
from collections import OrderedDict
import torch
import torchvision.models.vgg as vgg

LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])
new_state_dict = OrderedDict()

class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.features = vgg.vgg16(pretrained=False).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '30': "relu5",
        }
        # self.layer_name_mapping = {
        #     '14': "relu1",
        #     '17': "relu3",
        #
        # }

    def forward(self, x):
        output = {}
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
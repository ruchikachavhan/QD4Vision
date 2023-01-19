import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import resnet50, resnet18
from timm.models.convnext import convnext_base
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

class BranchedResNet(nn.Module):
    def __init__(self, N, arch, num_classes, stop_grad = True, pretrained=True):
        super(BranchedResNet, self).__init__()
        if arch == 'resnet50':
            self.base_model = resnet50(pretrained = pretrained)
            self.num_feat = 2048
        elif arch == 'resnet18':
            self.base_model = resnet18(pretrained = pretrained)
            self.num_feat = 512
        self.N = N
        self.num_classes = num_classes

        # Branching out only one Resnet50 layer
        self.base_model.branches_layer4 = nn.ModuleList([resnet50(pretrained=False).layer4 for _ in range(N)])
        self.base_model.branches_fc = nn.ModuleList([nn.Linear(self.num_feat, num_classes) for _ in range(N)])

        del self.base_model.layer4, self.base_model.fc

        if stop_grad:
            for name, param in self.base_model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

    def forward(self, x, reshape = True):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.act1(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        feats = [self.base_model.global_pool(self.base_model.branches_layer4[i](x)) for i in range(self.N)]
        outputs = [self.base_model.branches_fc[i](feats[i]) for i in range(self.N)]

        if reshape:
            outputs = torch.cat(outputs).reshape(self.N, -1, self.num_classes)
            feats = torch.cat(feats).reshape(self.N, -1, self.num_feat)

        return outputs, feats

# TODO: Residual Adapter models


# # Checks
# model = DiverseResNet(N = 5, arch ='resnet50', num_classes = 100)
# print(model)

# x = torch.randn((16, 3, 224, 224))
# feats, outputs = model(x)
# print(len(feats), len(outputs), feats[0].shape, outputs[0].shape)
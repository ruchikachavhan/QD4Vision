import copy
import torch.nn as nn
import torch
import numpy as np

def enable_grad(module, keep_grad):
    for p in module.parameters():
        keep = keep_grad
        # Use bool here - because of an error
        p.requires_grad = bool(keep)


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        state_dict=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.transforms = None
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)
        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        x = self.fc(feats)
        return x, feats

    def forward(self, x):
        return self._forward_impl(x)

class TSA_Conv2d(nn.Module):
    def __init__(self, op, num_adapters, ad_type, ad_form):
        # Here num_adapters is the number of adapters per conv 
        super().__init__()
        self.op = copy.deepcopy(op)
        # ad_type defines what kind of adapter to use: options - residual, series, channel-wise
        self.ad_type = ad_type
        self.ad_form = ad_form
        self.num_adapters = num_adapters
        if self.ad_type == 'residual':
            if self.ad_form == 'matrix':
                if op.stride[0] == 2:
                    self.alpha = nn.ModuleList([nn.Conv2d(self.op.out_channels, self.op.out_channels, kernel_size=3, stride= (2, 2), padding=(1,1), bias=True) for _ in range(num_adapters)])
                else:
                    self.alpha = nn.ModuleList([nn.Conv2d(self.op.out_channels, self.op.out_channels, kernel_size=1, bias=True) for _ in range(num_adapters)])
            else:
                # channelwise
                self.alpha = nn.ParameterList([nn.Parameter(torch.ones(1, self.op.out_channels, 1, 1)) for _ in range(num_adapters)])
                self.alpha_bias = nn.ParameterList([nn.Parameter(torch.ones(1, self.op.out_channels, 1, 1)) for _ in range(num_adapters)])

        elif self.ad_type == 'series':
            if self.ad_form == 'matrix':
                self.alpha = nn.ModuleList([nn.Conv2d(self.op.out_channels, self.op.out_channels, kernel_size=1, bias=True) for _ in range(num_adapters)])
            else:
                # channelwise
                self.alpha = nn.ParameterList([nn.Parameter(torch.ones(1, self.op.out_channels, 1, 1)) for _ in range(num_adapters)])
                self.alpha_bias = nn.ParameterList([nn.Parameter(torch.ones(1, self.op.out_channels, 1, 1)) for _ in range(num_adapters)])
        
        if self.ad_form == 'matrix':
            for i in range(self.num_adapters):
                nn.init.dirac_(self.alpha[i].weight)
        self.adapt = None
       
        
    def forward(self, x):
        assert self.adapt is not None  # if it is still None, we forgot to set it
        op_x = self.op(x)
        output = 0.0

        for k in range(len(self.alpha)):
            if self.adapt[k]:
                if self.ad_type == 'residual':
                    if self.ad_form == 'matrix':
                        output += self.alpha[k](x)
                    else:
                        output += x * self.alpha[k] + self.alpha_bias[k]
                elif self.ad_type == 'series':
                    if self.ad_form == 'matrix':
                        output += self.alpha[k](op_x)
                    else:
                        output += op_x * self.alpha[k] + self.alpha_bias[k]

        if np.all(self.adapt == 0):
            return op_x
        else:
            if self.ad_type == 'residual':
                return op_x + output
            elif self.ad_type == 'series':
                return output

class TSA_ResNet(ResNet):
    def __init__(
        self,
        block,
        layers,
        ad_type,
        ad_form,
        num_adapters=5,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        state_dict=None,
    ):
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            state_dict=state_dict,
        )
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                        setattr(block, name, TSA_Conv2d(m, num_adapters, ad_type, ad_form))

    def forward(self, x):
        return super().forward(x)

def sample_adapter_configuration(num_adapters):
    return np.random.rand(num_adapters).round().astype(np.uint8)

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())
     

class SuperNetSampler(nn.Module):
    def __init__(self, model, num_adapters, num_adapters_per_conv):
        super().__init__()
        self.model = model
        self.num_adapters = num_adapters
        self.num_adapters_per_conv = num_adapters_per_conv

    def sample_subnet(self, adapter_configuration=None):
        # optionally you can pass your own adapter configuration, instead of randomly sampling it
        if adapter_configuration is None:
            adapter_configuration = np.array([sample_adapter_configuration(self.num_adapters_per_conv) for _ in range(self.num_adapters)])
        disable_grad(self.model)
        i_conf = 0
        for _, m in self.model.named_modules():
            if isinstance(m, TSA_Conv2d):
                m.adapt = (adapter_configuration[i_conf, :] == 1)
                if m.ad_form == 'matrix':
                    for i in range(0, len(m.adapt)):
                        enable_grad(m.alpha[i], m.adapt[i])
                else:
                    for i in range(0, len(m.adapt)):
                        m.alpha[i].requires_grad = bool(m.adapt[i])
                        m.alpha_bias[i].requires_grad = bool(m.adapt[i])
                i_conf += 1

    def forward(self, x):
        return self.model(x)

def create_tsa_resnet(state_dict, block, layers, ad_type, ad_form, supernet=True, **kwargs):
    # can optionally pass a state_dict, if ResNet backbone is pre-trained
    model = TSA_ResNet(block, layers, ad_type = ad_type, ad_form = ad_form, state_dict=state_dict, **kwargs)
    if supernet:
        num_adapters = len([n for n, m in model.named_modules() if isinstance(m, TSA_Conv2d)])
        model = SuperNetSampler(model, num_adapters, num_adapters_per_conv=5)
    return model

# model = create_tsa_resnet(None, Bottleneck, [3, 4, 6, 3], True)
# model.sample_subnet()
# x = torch.randn(64, 3, 224, 224)
# output = model(x)
# print(output.shape)
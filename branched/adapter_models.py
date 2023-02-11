import torch
import torch.nn as  nn
import torch.nn.functional as F

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    
    def __init__(self, planes, mode, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        if mode == 'series_adapters':
            self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes))
        elif mode == 'parallel_adapters':
            self.conv = conv1x1_fonc(planes, out_planes, stride) 
        else:
            self.conv = conv1x1_fonc(planes)
        self.mode = mode
    def forward(self, x):
        y = self.conv(x)
        if self.mode == 'series_adapters':
            y += x
        return y

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, mode = 'parallel_adapters', num_tasks = 5, is_proj = 1, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        if mode == 'series_adapters' and is_proj:
            self.batch_norm2 = nn.ModuleList([nn.Sequential(conv1x1(out_channels, mode = mode), nn.BatchNorm2d(out_channels)) for i in range(num_tasks)])
        elif mode == 'parallel_adapters' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(out_channels, mode, out_channels, stride) for i in range(num_tasks)])
            self.batch_norm2 = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_tasks)])
        else:
            self.batch_norm2 = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_tasks)])

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        self.mode = mode
        self.is_proj = is_proj

        self.num_tasks = num_tasks
        
    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            identity = x[i].clone()
            x_i = self.relu(self.batch_norm1(self.conv1(x[i])))
            y_i = self.conv2(x_i)

            if self.mode == 'parallel_adapters' and self.is_proj:
                y_i = y_i + self.parallel_conv[i](x_i)
            y_i = self.batch_norm2[i](y_i)

            # x = self.relu(self.batch_norm2(self.conv2(x)))
        
            y_i = self.conv3(y_i)
            y_i = self.batch_norm3(y_i)
        
            #downsample if needed
            if self.i_downsample is not None:
                identity = self.i_downsample(identity)
            #add identity
            y_i += identity
            y_i = self.relu(y_i)
        
            outputs.append(y_i)

        return outputs

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()
      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      x += identity
      x = self.relu(x)
      return x

        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_tasks, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.ModuleList([nn.Linear(512*ResBlock.expansion, num_classes) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.num_classes = num_classes

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x_list = [x for _ in range(self.num_tasks)]
        # print("Inputs list", len(x_list))
        x = self.layer1(x_list)
        # 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        feats, outputs = [], []
        for i in range(self.num_tasks):
            x_i = self.avgpool(x[i])
            feats.append(x_i)
            x_i = x_i.reshape(x_i.shape[0], -1)
            out = self.fc[i](x_i)
            outputs.append(out)

        outputs = torch.cat(outputs).reshape(self.num_tasks, -1, self.num_classes)
        feats = torch.cat(feats).reshape(self.num_tasks, -1, 2048)
        return outputs, feats
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, num_tasks, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, num_tasks, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)


# model = ResNet50(num_classes= 100, num_tasks = 5)
# x = torch.randn(16, 3, 224, 224)
# print(model)
# model(x)
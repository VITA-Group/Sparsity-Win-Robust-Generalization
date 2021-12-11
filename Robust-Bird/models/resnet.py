import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 



#################### resnet definition ##################################################
class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
	    """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
		"""
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.select = channel_selection(inplanes)
        self.conv1 = conv3x3(cfg[0], cfg[1], stride)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.select(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, seed, num_classes=10, cfg=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        rng = torch.manual_seed(seed)
        if cfg is None:
            # Construct config variable (basic block)
            cfg = [[64], [64, 64]*layers[0], [128, 128]*layers[1], [256, 256]*layers[2], [512, 512]* layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[0:2*layers[0]])
        self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[2*layers[0]: 2*(layers[0]+layers[1])], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[2*(layers[0]+layers[1]): 2*(layers[0]+layers[1]+layers[2])], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[2*(layers[0]+layers[1]+layers[2]): 2*(layers[0]+layers[1]+layers[2]+layers[3])], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.select = channel_selection(64 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n), generator=rng)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01, generator=rng)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # make it suitable for 64 * 64 input (TinyImageNet)
        latent = x.view(x.size(0), -1)
        y = self.fc(latent)
        if with_latent:
            return y, latent
        return y


def resnet18(pretrained=False, cfg=None, num_classes=10, seed=0, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], seed, num_classes=num_classes, cfg=cfg, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model

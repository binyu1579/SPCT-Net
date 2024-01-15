import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class ResNeXtBottleneck(nn.Cell):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            pad_mode='pad',
            padding=1,
            group=cardinality,
            has_bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNeXt3D(nn.Cell):
    def __init__(self, block, layers, shortcut_type='B', cardinality=32):
        self.inplanes = 32
        super(ResNeXt3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, cardinality, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, cardinality, stride=(2, 2, 1))
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, cardinality, stride=(2, 2, 2))

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=(1, 1, 1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = P.AvgPool3D(kernel_size=1, stride=stride)
            else:
                downsample = nn.SequentialCell([
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        has_bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                ])

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.SequentialCell(layers)

    def construct(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features


if __name__ == '__main__':
    # 示例输入
    input_data = Tensor(np.random.randn(1, 1, 160, 160, 48).astype(np.float32))

    # 创建MindSpore模型
    ms_model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3])
    output = ms_model(input_data)
    for y in output:
        print(y.shape)

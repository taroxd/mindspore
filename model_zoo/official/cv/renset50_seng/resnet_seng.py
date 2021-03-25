import math
import numpy as np
import mindspore as ms

from seng_layer import Conv2d_SENG_GPU, Dense_SENG_GPU


def _conv_init_kaiming_normal(shape):
    # mode='fan_out', nonlinearity='relu', a=0
    assert len(shape)==4
    fan = shape[0] * shape[2] * shape[3]
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    ret = ms.Tensor(np.random.normal(0, std, size=shape), dtype=ms.float32)
    return ret


def _dense_init_kaiming_uniform(shape):
    # mode='fan_in', a=math.sqrt(5), nonlinearity='leaky_relu'.....strange
    assert len(shape)==2
    fan = shape[1]
    gain = math.sqrt(2 / (1+5))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3) * std  # Calculate uniform bounds from standard deviation
    ret = ms.Tensor(np.random.uniform(-bound, bound, size=shape), dtype=ms.float32)
    return ret


def _conv1x1(in_channel, out_channel, stride, damping, input_hw, extra_args):
    weight = _conv_init_kaiming_normal((out_channel, in_channel, 1, 1))
    layer = Conv2d_SENG_GPU(in_channel, out_channel,
                            kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                            damping=damping, input_hw=input_hw, extra_args=extra_args)
    return layer


def _conv3x3(in_channel, out_channel, stride, damping, input_hw, extra_args):
    weight = _conv_init_kaiming_normal((out_channel, in_channel, 3, 3))
    layer = Conv2d_SENG_GPU(in_channel, out_channel,
                            kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                            damping=damping, input_hw=input_hw, extra_args=extra_args)
    return layer


def _conv7x7(in_channel, out_channel, stride, damping, input_hw, extra_args):
    weight = _conv_init_kaiming_normal((out_channel, in_channel, 7, 7))
    layer = Conv2d_SENG_GPU(in_channel, out_channel,
                            kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                            damping=damping, input_hw=input_hw, extra_args=extra_args)
    return layer


def _bn(channel):
    return ms.nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return ms.nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, damping, extra_args):
    weight = _dense_init_kaiming_uniform((out_channel, in_channel))
    layer = Dense_SENG_GPU(in_channel, out_channel, has_bias=False, weight_init=weight,
                            bias_init=0, damping=damping, activation=None, extra_args=extra_args)
    return layer


class ResidualBlock(ms.nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride,
                 damping,
                 input_hw,
                 extra_args):
        super().__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, damping=damping, input_hw=input_hw, extra_args=extra_args)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride, damping=damping, input_hw=self.conv1.infer_out_hw(), extra_args=extra_args)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, damping=damping, input_hw=self.conv2.infer_out_hw(), extra_args=extra_args)
        self.bn3 = _bn_last(out_channel)

        self.relu = ms.nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = ms.nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                 damping=damping, input_hw=input_hw, extra_args=extra_args),
                                                        _bn(out_channel)])
        self.output_hw = self.conv3.infer_out_hw()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.relu(out + identity)

        return out


class ResNet(ms.nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 damping,
                 input_hw,
                 tag_cifar,
                 extra_args):
        super().__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.tag_cifar = tag_cifar
        if self.tag_cifar:
            self.conv1 = _conv3x3(3, 64, stride=1, damping=damping, input_hw=input_hw, extra_args=extra_args)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, damping=damping, input_hw=input_hw, extra_args=extra_args)
        self.bn1 = _bn(64)
        self.relu = ms.ops.ReLU()

        self.maxpool = ms.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        tmp0 = self.conv1.infer_out_hw()
        if not tag_cifar:
            tmp0 = tmp0[0]//2, tmp0[1]//2
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       damping=damping,
                                       input_hw=tmp0,
                                       extra_args=extra_args)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       damping=damping,
                                       input_hw=self.layer1.output_hw,
                                       extra_args=extra_args)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2], damping=damping,
                                       input_hw=self.layer2.output_hw,
                                       extra_args=extra_args)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       damping=damping,
                                       input_hw=self.layer3.output_hw,
                                       extra_args=extra_args)
        self.mean = ms.ops.ReduceMean(keep_dims=True)
        self.flatten = ms.nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, damping=damping, extra_args=extra_args)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, damping, input_hw, extra_args):
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride,
                             damping=damping, input_hw=input_hw, extra_args=extra_args)
        layers.append(resnet_block)
        input_hw = resnet_block.output_hw

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1,
                                 damping=damping, input_hw=input_hw, extra_args=extra_args)
            layers.append(resnet_block)
            input_hw = resnet_block.output_hw

        layers_cell = ms.nn.SequentialCell(layers)
        layers_cell.output_hw = input_hw
        return layers_cell

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.tag_cifar:
            c1 = x
        else:
            c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def resnet50(class_num=10, damping=0.03, input_hw=(32, 32), extra_args=None, tag_cifar=None):
    if tag_cifar is None:
        tag_cifar = input_hw[0]<=32
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  damping,
                  input_hw,
                  tag_cifar,
                  extra_args)

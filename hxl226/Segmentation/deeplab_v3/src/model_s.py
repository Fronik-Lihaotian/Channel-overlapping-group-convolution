from torch import nn
import torch
from torch import Tensor


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def s_pointwise_two_ex(x: Tensor):
    batch_size, num_channels, height, width = x.size()
    if num_channels % 4 == 0:
        x_a, x_b, x_c, x_d = x.chunk(4, dim=1)
        return x_a, x_b, x_c, x_d
    else:
        return x


def s_pointwise_two_reex(x: Tensor):
    batch_size, num_channels, height, width = x.size()
    if num_channels % 4 == 0:
        x_a, x_b, x_c, x_d = x.chunk(4, dim=1)
        # x1 = torch.cat((x_c, x_d), dim=1)
        # x2 = torch.cat((x_d, x_a), dim=1)
        # x3 = torch.cat((x_a, x_b), dim=1)
        # x4 = torch.cat((x_b, x_c), dim=1)
        x = torch.cat((x_a, x_b, x_b, x_c, x_c, x_d, x_d, x_a), dim=1)
        return x
    else:
        return x


def s_pointwise_three_ex(x: Tensor):
    batch_size, num_channels, height, width = x.size()
    # if num_channels % 4 == 0:
    x_a, x_b, x_c, x_d = x.chunk(4, dim=1)
    x1 = torch.cat((x_a, x_b, x_c), dim=1)
    x2 = torch.cat((x_b, x_c, x_d), dim=1)
    x3 = torch.cat((x_c, x_d, x_a), dim=1)
    x4 = torch.cat((x_d, x_a, x_b), dim=1)
    return x1, x2, x3, x4
    # else:
    #     return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
        )
        self.out_channels = out_channel


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        # self.conv5 = nn.Sequential(nn.Conv2d(in_channel // 2, hidden_channel // 4, kernel_size=1, bias=False)
        #                            # nn.ReLU(inplace=True)
        #                            )
        # self.conv6 = nn.Sequential(nn.Conv2d(in_channel // 2, hidden_channel // 4, kernel_size=1, bias=False))
        # self.conv7 = nn.Sequential(nn.Conv2d(in_channel // 2, hidden_channel // 4, kernel_size=1, bias=False))
        # self.conv8 = nn.Sequential(nn.Conv2d(in_channel // 2, hidden_channel // 4, kernel_size=1, bias=False))
        self.pointsign = False
        self.use_shortcut = stride == 1 and in_channel == out_channel
        # layers_A = []
        layers = []
        # layers_C = []
        # layers_D = []
        # if expand_ratio != 1:
        #     # 1x1 pointwise conv
        #     self.pointsign = True
        layers.append(
            nn.Sequential(nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=1, groups=4, bias=False),
                          nn.BatchNorm2d(hidden_channel)))
            # layers_A.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
            # layers_A.append(nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=1, groups=4, bias=False))

        # layers_A.extend([
        #     nn.BatchNorm2d(hidden_channel),
        #     # nn.SiLU(),
        #     # nn.Hardswish(inplace=True),
        #     # 3x3 depthwise conv
        #     # ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
        #     nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=hidden_channel, bias=False),
        #     nn.BatchNorm2d(hidden_channel),
        #     nn.SiLU(),
        #     # 1x1 pointwise conv(linear)
        #     nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
        #     # nn.Conv2d(hidden_channel, out_channel, kernel_size=1, groups=4, bias=False),
        #     nn.BatchNorm2d(out_channel),
        # ])
        layers.extend([

            # nn.SiLU(),
            # nn.Hardswish(inplace=True),
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=hidden_channel,
            #           bias=False),
            # nn.BatchNorm2d(hidden_channel),
            # nn.SiLU(),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # nn.Conv2d(hidden_channel, out_channel, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        # layers_B.extend([
        #     nn.Conv2d(hidden_channel // 2, out_channel // 4, kernel_size=1, bias=False),
        #     # nn.BatchNorm2d(int(out_channel/4)),
        # ])
        # layers_C.extend([
        #     nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channel),
        # ])
        # layers_D.extend([
        #     nn.BatchNorm2d(out_channel),
        # ])

        # self.conv1 = nn.Sequential(*layers_A)
        self.conv = nn.Sequential(*layers)
        # self.conv3 = nn.Sequential(*layers_C)
        # self.conv4 = nn.Sequential(*layers_D)
        # self.conv6 = nn.Sequential(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        # self.conv7 = nn.Sequential(nn.BatchNorm2d(hidden_channel),
        #                            # nn.LeakyReLU(inplace=True),
        #                            )
        self.out_channels = out_channel
        self.is_strided = stride > 1

    def forward(self, x):
        # if self.use_shortcut:
        # out_a, out_b, out_c, out_d = s_pointwise_two_ex(x)
        # out_l = []
        # out_l.append(self.conv5(torch.cat((out_a, out_b), dim=1)))
        # out_l.append(self.conv6(torch.cat((out_b, out_c), dim=1)))
        # out_l.append(self.conv7(torch.cat((out_c, out_d), dim=1)))
        # out_l.append(self.conv8(torch.cat((out_d, out_a), dim=1)))
        # for o in out:
        #     out_l.append(self.conv5(o))
        # print(self.conv5[0].weight)
        # x_b = torch.cat((out_l[0], out_l[1], out_l[2], out_l[3]),
        #                 dim=1)
        # x_b = self.conv7(x_b)
        # else:
        #     x_b = x

        if self.use_shortcut:
            return x + self.conv(s_pointwise_two_reex(x))
        else:
            # if self.pointsign:
            return self.conv(s_pointwise_two_reex(x))
            # else:
            #     return self.conv2(x)

        #     out = self.conv1(x_b)
        #     if len(S_pointwise_two_ex(out)) > 1:
        #         out_a, out_b, out_c, out_d = S_pointwise_two_ex(out)
        #         x_p = torch.cat((self.conv2(out_a), self.conv2(out_b),
        #                         self.conv2(out_c), self.conv2(out_d)),
        #                         dim=1)
        #         return x + self.conv4(x_p)
        #     else:
        #         return x + self.conv3(out)
        # else:
        #     x = self.conv6(x)
        #     out = self.conv1(x)
        #     return self.conv3(out)
        # if self.use_shortcut:
        #     return x + self.conv1(x)
        # else:
        #     return self.conv1(x)


class MobileNet_s(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNet_s, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # inverted_residual_setting = [
        #     # t, c, n, s
        #     # [1, 16, 1, 1],
        #     [4, 24, 2, 2],
        #     [4, 32, 3, 2],
        #     [4, 64, 4, 2],
        #     [4, 96, 3, 1],
        #     [4, 144, 3, 2],
        #     [4, 216, 2, 1],
        #     [4, 324, 1, 1],
        # ]
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # print(self.features[1].conv5[0].weight)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

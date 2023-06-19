import torch
import torch.nn as nn

from torch.nn import functional as F
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)) # inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # math.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s        
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],     # stride = 2, layer -> 1
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],     # stride = 2, layer -> 3
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],     # stride = 2, layer -> 6
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],     # stride = 2, layer -> 12
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)



def mobilenet_v3_module(pretrain=True, freeze=True):
    """Constructs a mobilenet_v3 model.
    """
    model = mobilenetv3_large()
    if pretrain:
        model.load_state_dict(torch.load('utils/model/mobilenetv3-large-1cd25616.pth'))
 
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


class textEcoding(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel):
        super(textEcoding, self).__init__()

        self.TE1 = nn.Sequential(
                    nn.Conv2d(input_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False), 
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU(inplace=True)
                    )

        self.TE2 = nn.Sequential(
                    nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False, dilation=1), 
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU(inplace=True)
                    )

        self.TE3 = nn.Sequential(
                    nn.Conv2d(mid_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), 
                    nn.BatchNorm2d(output_channel),
                    nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.TE1(x)
        x = self.TE2(x)
        out = self.TE3(x)
        return out


class encode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mbnet_layer1 = nn.Sequential(*(list(model.children())[0][:4])) # 0~3 x 4
        self.mbnet_layer2 = nn.Sequential(*(list(model.children())[0][4:7])) # 4~6 x 2
        self.mbnet_layer3 = nn.Sequential(*(list(model.children())[0][7:13])) # 7~12 x 2
        self.mbnet_layer4 = nn.Sequential(*(list(model.children())[0][13:]),
                                *(list(model.children())[1])) # 0: (13~15) + 1: (*) x 2

    def forward(self, x):
        x1 = self.mbnet_layer1(x)
        x2 = self.mbnet_layer2(x1)
        x3 = self.mbnet_layer3(x2)
        x4 = self.mbnet_layer4(x3)
        return x1, x2, x3, x4 # x1 -> 24, x2 -> 40, x3 -> 112, x4 -> 960channels


##################### Final model assembly #####################
class mosts(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = encode(model=mobilenet_v3_module(pretrain=True, freeze=True))

        self.encode_texture = textEcoding(960, 256, 1024)
        self.encode_texture1 = textEcoding(960, 256, 1024)

        self.ca = ChannelAttention(in_planes=1024)

        self.avg_pool1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.cos_similarity_func = nn.CosineSimilarity()
        
        self.embedding = nn.Sequential( # depth - point - wise convolution
            nn.Conv2d(512, 512, groups = 512, kernel_size = 3, stride = 1, dilation = 1, padding = 1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
            )
 
        self.decode2 = nn.Sequential(
            nn.Conv2d(1136, 512, 1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(552, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(280, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1) 
        )

    def forward(self, image, patch):
        img1, img2, img3, img4 = self.encode(image)     # backbone encoding 24 40 112 960
        _, _, _, pat4 = self.encode(patch)              # backbone encoding 24 40 112 960

        img4 = self.encode_texture(img4)            # Texture encoding
        pat4 = self.encode_texture1(pat4)           # Texture encoding
      
        similarity = self.cos_similarity_func(img4, self.avg_pool1x1(pat4)).unsqueeze(dim=1) 
        img_seg = similarity * img4 
      
        ch_split = img_seg.size(dim=1) // 4
        for i in range(4):
            seg = img_seg[:, ch_split * i:ch_split * i + ch_split, :, :]
            ori = img4[:, ch_split * i:ch_split * i + ch_split, :, :]

            img_g = torch.cat([seg, ori], dim=1)
            img_g = self.embedding(img_g)
            ca = self.ca(img_g) # Channel-Wise Attention
            img_g = ca * img_g
            if i == 0:
                img = img_g
            else:
                img += img_g

        # Decoder
        img = F.interpolate(img, 16, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img3], dim=1)
        img = self.decode2(img)

        img = F.interpolate(img, 32, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img2], dim=1)
        img = self.decode3(img)

        img = F.interpolate(img, 64, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img1], dim=1)
        img = self.decode4(img)

        img = F.interpolate(img, 256, mode='bilinear', align_corners=False) # bilinear upsampling x 4

        img = torch.sigmoid(img)
        return img
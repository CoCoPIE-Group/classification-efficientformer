import numpy as np
import os
import torch
import torch.nn as nn
from timm.models.registry import register_model

def _make_divisible(v, divisor=4, min_value=None,scaling_factor=0.9):
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
    new_v = scaling_factor * v
    new_v = max(min_value, int(new_v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class AdaptiveInvertedResidual(nn.Module):
    def __init__(self, residual_settings, stride, t,scaling_factor=0.9):
        super(AdaptiveInvertedResidual, self).__init__()

        assert stride in [1, 2]
        self.stride = stride
        self.t = t
        self.scaling_factor =scaling_factor
        # self.cnt = cnt
        self.relu = nn.ReLU6(inplace=True)

        if t != 1:
            conv1_in_ch, conv1_out_ch = residual_settings['conv1']
            conv1_in_ch = _make_divisible(conv1_in_ch,scaling_factor=self.scaling_factor)
            conv1_out_ch = _make_divisible(conv1_out_ch,scaling_factor=self.scaling_factor)
            self.conv1 = nn.Conv2d(conv1_in_ch, conv1_out_ch, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = residual_settings['conv2']
        conv2_in_ch = _make_divisible(conv2_in_ch,scaling_factor=self.scaling_factor)
        conv2_out_ch = _make_divisible(conv2_out_ch,scaling_factor=self.scaling_factor)
        self.conv2 = nn.Conv2d(conv2_in_ch, conv2_out_ch, 3, stride, 1, groups=conv2_in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_out_ch)

        conv3_in_ch, conv3_out_ch = residual_settings['conv3']
        conv3_in_ch = _make_divisible(conv3_in_ch,scaling_factor=self.scaling_factor)
        conv3_out_ch = _make_divisible(conv3_out_ch,scaling_factor=self.scaling_factor)
        self.conv3 = nn.Conv2d(conv3_in_ch, conv3_out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_out_ch)

        self.inplanes = conv1_in_ch if t != 1 else conv2_in_ch
        self.outplanes = conv3_out_ch

    def forward(self, x):
        residual = x

        if self.t != 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = x

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.cnt == 17:
        #     out *= 0.

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class AdaptiveMobileNetV2(nn.Module):
    def __init__(self, ch_cfg, num_classes=1000, input_size=224,scaling_factor=0.9):
        super(AdaptiveMobileNetV2, self).__init__()

        channels = np.load(ch_cfg, allow_pickle=True).item()
        self.scaling_factor = scaling_factor
        self.num_classes = num_classes
        self.relu = nn.ReLU6(inplace=True)

        conv1_out_ch = channels['conv1']
        conv1_out_ch = _make_divisible(conv1_out_ch,scaling_factor=self.scaling_factor)
        self.conv1 = nn.Conv2d(3, conv1_out_ch, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        self.blocks = self._make_blocks(channels)

        conv_last_in_ch, conv_last_out_ch = channels['conv_last']
        conv_last_in_ch = _make_divisible(conv_last_in_ch,scaling_factor=self.scaling_factor)
        conv_last_out_ch = _make_divisible(conv_last_out_ch,scaling_factor=self.scaling_factor)
        self.conv_last = nn.Conv2d(conv_last_in_ch, conv_last_out_ch, 1,
                                   bias=False)
        self.bn_last = nn.BatchNorm2d(conv_last_out_ch)
        self.avgpool = nn.AvgPool2d(input_size // 32)
        self.fc = nn.Conv2d(conv_last_out_ch, self.num_classes, 1)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(x).view(x.size(0), -1)

        if self.training:
            return x,x
        else:
            return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_blocks(self, residual_settings):
        blocks_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            # [6, 64, 3, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # for cifar-10
        if self.num_classes == 10:
            blocks_setting[2] = [6, 24, 2, 1]

        blocks = []
        cnt = 0
        for t, c, n, s in blocks_setting:
            for i in range(n):
                if True:
                    blocks.append(AdaptiveInvertedResidual(residual_settings[str(cnt)], s if i == 0 else 1, t,scaling_factor=self.scaling_factor))
                cnt += 1

        return nn.Sequential(*blocks)

@register_model
def adaptive_mobilenet_v2(mobilenet_config_path=None, num_classes=1000, input_size=224,scaling_factor=0.9,**kwargs):
    assert mobilenet_config_path is not None, ('mobilenet_config_path cannot be empty ')
    return AdaptiveMobileNetV2(mobilenet_config_path, num_classes, input_size,scaling_factor)


class ProfileConv(nn.Module):
    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []

        def hook_conv(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            # self.macs.append(module.weight.size(0) * module.weight.size(1) *
            #                  module.weight.size(-1) * module.weight.size(-1))

        def hook_linear(module, input, output):
            # self.macs.append(output.size(1) * output.size(2) * output.size(3) *
            #                  module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            self.macs.append(module.weight.size(0) * module.weight.size(1))

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(hook_linear))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        return self.macs


if __name__ == '__main__':

    x = torch.randn(1, 3, 224, 224)

    model = adaptive_mobilenet_v2('/home/guanhua/gding/classification-efficientformer/mobilenet_config/mbv2_base_config.npy',scaling_factor = 0.94)

    # checkpoint = torch.load('./mbv2-211m/checkpoints/retrain.pth', map_location='cpu')['model']
    #
    # new_ckpt = {}
    # for item in checkpoint:
    #     new_ckpt[item[7:]] = checkpoint[item]
    #
    # model.load_state_dict(new_ckpt)

    print(model)

    checkpoint = torch.load("/home/guanhua/gding/classification-efficientformer/mbnet_192m.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()

    # torch.onnx.export(model, x, "mobilenet_192M.onnx", verbose=True)
    # print('onnx exported')


    # from xgen_tools.profile import _profile
    # from thop import clever_format
    # import copy
    #
    # shape = (1, 3, 244, 244)
    # model_copy = copy.deepcopy(model)
    # device = next(model_copy.parameters()).device
    # x = torch.randn(shape).to(device)
    # flops, params = _profile(model_copy, inputs=(x,), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")


    profile = ProfileConv(model)
    MACs = profile(torch.randn(1, 3, 224, 224))
    print(len(MACs))
    print(sum(MACs) / 1e9, 'GMACs, only consider conv layers')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections
import vgg16_c
import resnet
import random
from operator import add
from functools import reduce

class MCCCN(nn.Module):
    def __init__(self, backbone_name='resnet101', batch_norm=False, load_weights=False,device = 'cuda:0'):
        super(MCCCN, self).__init__()
        self.backbone_name = backbone_name
        self.device = device
        if 'vgg16' in self.backbone_name:
            self.backbone = vgg16_c.VGG16_C(pretrain, logger)
            self.features = self.getVGGFeature_List
        elif 'resnet101' in self.backbone_name:
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 23, 3]
            self.features = self.getResNetFeature_List
        elif self.backbone_net_name == 'resnext-101':
            self.backbone = resnet.resnext101_32x8d(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 23, 3]
            self.features = self.getResNetFeature_List
        t=2
        self.conv0_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        t=8
        # change channel to 21 which is easy to cal
        self.conv1_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv1_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        t=16
        self.conv2_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv2_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv2_4_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        t=32
        self.conv3_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_4_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_5_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_6_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_7_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_8_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_9_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_10_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_11_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_12_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_13_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_14_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_15_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_16_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_17_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_18_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_19_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_20_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_21_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_22_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_23_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        t=64
        self.conv4_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)

        # upsample
        self.feature_upsample_4 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1, bias=False)
        self.feature_upsample_3 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1, bias=False)
        self.feature_upsample_2 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1, bias=False)
        self.feature_upsample_1 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1, bias=False)
        self.feature_upsample_0 = nn.ConvTranspose2d(21, 1, 4, stride=2, padding=1, bias=False)
        
        self.conv4_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)
        self.conv3_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)
        self.conv2_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)
        self.conv1_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)
        self.conv0_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)

    def getVGGFeature_List(self, img, device='cuda:0'):
        return self.backbone(img)

    def getResNetFeature_List(self, img, device='cuda:0'):
        feats = []
        nbottlenecks = self.nbottlenecks
        bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        # print(self.bottleneck_ids) [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1, 2]
        layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        # print(self.layer_ids) [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4]

        # Layer 0
        feat = self.backbone.conv1.forward(img)

        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)

        feats.append(feat.clone())
        feat = self.backbone.maxpool.forward(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(bottleneck_ids, layer_ids)):

            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feats.append(feat.clone())

        return feats
    def calLayer0(self, feats):
        sum0 = self.conv0_down(feats[0])
        return sum0
    def calLayer1(self, feats):
        sum1 = self.conv1_1_down(feats[1]) + \
               self.conv1_2_down(feats[2]) + \
               self.conv1_3_down(feats[3])
        return sum1

    def calLayer2(self, feats):
        sum2 = self.conv2_1_down(feats[4]) + \
               self.conv2_2_down(feats[5]) + \
               self.conv2_3_down(feats[6]) + \
               self.conv2_4_down(feats[7])
        return sum2

    def _vgg_calLayer3(self, feats):
        sum3 = self.conv3_1_down(feats[8]) + \
               self.conv3_2_down(feats[9]) + \
               self.conv3_3_down(feats[10]) + \
               self.conv3_4_down(feats[11]) + \
               self.conv3_5_down(feats[12]) + \
               self.conv3_6_down(feats[13])
        return sum3

    def calLayer3(self, feats):
        sum3 = self.conv3_1_down(feats[8]) + \
               self.conv3_2_down(feats[9]) + \
               self.conv3_3_down(feats[10]) + \
               self.conv3_4_down(feats[11]) + \
               self.conv3_5_down(feats[12]) + \
               self.conv3_6_down(feats[13]) + \
               self.conv3_7_down(feats[14]) + \
               self.conv3_8_down(feats[15]) + \
               self.conv3_9_down(feats[16]) + \
               self.conv3_10_down(feats[17]) + \
               self.conv3_11_down(feats[18]) + \
               self.conv3_12_down(feats[19]) + \
               self.conv3_13_down(feats[20]) + \
               self.conv3_14_down(feats[21]) + \
               self.conv3_15_down(feats[22]) + \
               self.conv3_16_down(feats[23]) + \
               self.conv3_17_down(feats[24]) + \
               self.conv3_18_down(feats[25]) + \
               self.conv3_19_down(feats[26]) + \
               self.conv3_20_down(feats[27]) + \
               self.conv3_21_down(feats[28]) + \
               self.conv3_22_down(feats[29]) + \
               self.conv3_23_down(feats[30])
        return sum3

    def _vgg_calLayer4(self, feats):
        sum4 = self.conv4_1_down(feats[14]) + \
               self.conv4_2_down(feats[15]) + \
               self.conv4_3_down(feats[16])
        return sum4

    def calLayer4(self, feats):
        sum4 = self.conv4_1_down(feats[31]) + \
               self.conv4_2_down(feats[32]) + \
               self.conv4_3_down(feats[33])
        return sum4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.features(x, device=self.device)
        if self.backbone_name in ["resnet101", "resnext-101"]:
            calLayer4 = self.calLayer4
            calLayer3 = self.calLayer3
        elif self.backbone_name == "vgg16":
            calLayer4 = self._vgg_calLayer4
            calLayer3 = self._vgg_calLayer3
        calLayer2 = self.calLayer2
        calLayer1 = self.calLayer1
        calLayer0 = self.calLayer0
        
        sum0 = calLayer0(feats)
        sum1 = calLayer1(feats)
        sum2 = calLayer2(feats)
        sum3 = calLayer3(feats)
        sum4 = calLayer4(feats)
        
        sum4_upsamples = self.feature_upsample_4(sum4)
        sum3 = self.conv3_scale(torch.cat((sum3, sum4_upsamples), 1))
        sum3_upsamples = self.feature_upsample_3(sum3)
        sum2 = self.conv2_scale(torch.cat((sum2, sum3_upsamples), 1))
        sum2_upsamples = self.feature_upsample_2(sum2)
        sum1 = self.conv1_scale(torch.cat((sum1, sum2_upsamples), 1))
        sum1_upsamples = self.feature_upsample_1(sum1)
        sum0 = self.conv0_scale(torch.cat((sum0, sum1_upsamples), 1))
        sum0_upsamples = self.feature_upsample_0(sum0)
        
        return sum0_upsamples


if __name__ == '__main__':
    device = 'cuda:0'
    input_demo = torch.rand((2, 3, 224, 224)).to(device)
    target_demo = torch.rand((2, 1, 224, 224)).to(device)

    model = MCCCN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.95, weight_decay=5e-4)

    output_demo = model(input_demo)
    print(output_demo.shape)

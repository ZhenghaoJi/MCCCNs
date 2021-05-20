import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections
from . import resnet
from torchvision import models
import random
from operator import add
from functools import reduce
class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()          
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 64, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        
        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
class MCCCN_vgg(nn.Module):
    def __init__(self, backbone_name='vgg16', batch_norm=False, load_weights=False,device = 'cuda:0'):
        super(MCCCN_vgg, self).__init__()
        self.backbone_name = backbone_name
        self.device = device
        if 'vgg16' in self.backbone_name:
            self.features = self.getVGGFeature_List
            vgg = models.vgg16_bn(pretrained=True)
            features = list(vgg.features.children())
                # get each stage of the backbone
            self.features1 = nn.Sequential(*features[0:6])
            self.features2 = nn.Sequential(*features[6:13])
            self.features3 = nn.Sequential(*features[13:23])
            self.features4 = nn.Sequential(*features[23:33])
            self.features5 = nn.Sequential(*features[33:43])
        elif 'resnet101' in self.backbone_name:
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 23, 3]
            self.features = self.getResNetFeature_List
        elif self.backbone_net_name == 'resnext-101':
            self.backbone = resnet.resnext101_32x8d(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 23, 3]
            self.features = self.getResNetFeature_List
        t=2
        out=2
        rate=4
        self.conv1_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv2_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv4_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv5_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.msblock1_1 = MSBlock(64, rate)
        self.msblock2_1 = MSBlock(128, rate)
        self.msblock3_1 = MSBlock(256, rate)
        self.msblock4_1 = MSBlock(512, rate)
        self.msblock5_1 = MSBlock(512, rate)
        '''
        self.conv0_down = nn.Conv2d(32*t, 32*out, (1, 1), stride=1)
        #t=2
        # change channel to 32*out which is easy to cal
        self.conv1_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv1_3_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        #t=2
        self.conv2_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv2_3_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv2_4_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        #t=2
        self.conv3_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_4_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_5_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_6_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_7_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_8_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_9_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_10_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_11_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_12_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_13_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_14_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_15_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_16_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_17_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_18_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_19_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_20_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_21_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_22_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv3_23_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)

        self.conv4_1_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32 * t, 32*out, (1, 1), stride=1)
        '''
        # upsample
        
        '''
        self.msblock0 = MSBlock(64,rate)        
        self.msblock1_1 = MSBlock(256, rate)
        self.msblock1_2 = MSBlock(256, rate)
        self.msblock1_3 = MSBlock(256, rate)
        self.msblock2_1 = MSBlock(512, rate)
        self.msblock2_2 = MSBlock(512, rate)
        self.msblock2_3 = MSBlock(512, rate) 
        self.msblock2_4 = MSBlock(512, rate)
        self.msblock3_1 = MSBlock(1024, rate)
        self.msblock3_2 = MSBlock(1024, rate)
        self.msblock3_3 = MSBlock(1024, rate)
        self.msblock3_4 = MSBlock(1024, rate)
        self.msblock3_5 = MSBlock(1024, rate)
        self.msblock3_6 = MSBlock(1024, rate)
        self.msblock3_7 = MSBlock(1024, rate)
        self.msblock3_8 = MSBlock(1024, rate)
        self.msblock3_9 = MSBlock(1024, rate)
        self.msblock3_10 = MSBlock(1024, rate)
        self.msblock3_11 = MSBlock(1024, rate)
        self.msblock3_12 = MSBlock(1024, rate)
        self.msblock3_13 = MSBlock(1024, rate)
        self.msblock3_14 = MSBlock(1024, rate)
        self.msblock3_15 = MSBlock(1024, rate)
        self.msblock3_16 = MSBlock(1024, rate)
        self.msblock3_17 = MSBlock(1024, rate)
        self.msblock3_18 = MSBlock(1024, rate)
        self.msblock3_19 = MSBlock(1024, rate)
        self.msblock3_20 = MSBlock(1024, rate)
        self.msblock3_21 = MSBlock(1024, rate)
        self.msblock3_22 = MSBlock(1024, rate)
        self.msblock3_23 = MSBlock(1024, rate)
        self.msblock4_1 = MSBlock(2048, rate)
        self.msblock4_2 = MSBlock(2048, rate)
        self.msblock4_3 = MSBlock(2048, rate)
        '''

        self.conv4_scale = nn.Conv2d(out*32*2, out*32, (3, 3), stride=1, padding=1)
        self.conv3_scale = nn.Conv2d(out*32*2, out*32, (3, 3), stride=1, padding=1)
        self.conv2_scale = nn.Conv2d(out*32*2, out*32, (3, 3), stride=1, padding=1)
        self.conv1_scale = nn.Conv2d(out*32*2, 32*out, (3, 3), stride=1, padding=1)
        self.conv0_scale = nn.Conv2d(out*32, 1, (1, 1))
        
        self.density_head4 = nn.Sequential(
            MSBlock(64, rate),
            nn.Conv2d(out*32, 1, (1, 1))
        )

        self.density_head3 = nn.Sequential(
            MSBlock(64, rate),
            nn.Conv2d(out*32, 1, (1, 1))
        )

        self.density_head2 = nn.Sequential(
            MSBlock(64, rate),
            nn.Conv2d(out*32, 1, (1, 1))
        )

        self.density_head1 = nn.Sequential(
            MSBlock(64, rate),
            nn.Conv2d(out*32, 1, (1, 1))
        )
        
        self.confidence_head4 = nn.Sequential(
            nn.Conv2d(32*t, 32*t, 3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(32 * t, 1, (1, 1), stride=1)
        )

        self.confidence_head3 = nn.Sequential(
            nn.Conv2d(32*t, 32*t, 3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(32 * t, 1, (1, 1), stride=1)
        )

        self.confidence_head2 = nn.Sequential(
            nn.Conv2d(32*t, 32*t, 3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(32 * t, 1, (1, 1), stride=1)
        )

        self.confidence_head1 = nn.Sequential(
            nn.Conv2d(32*t, 32*t, 3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(32 * t, 1, (1, 1), stride=1)
        )       

    def getVGGFeature_List(self, img, device='cuda:0'):
        feats = []
        x1 = self.features1(img)
        feats.append(x1.clone())
        x2 = self.features2(x1)
        feats.append(x2.clone())
        x3 = self.features3(x2)
        feats.append(x3.clone())
        x4 = self.features4(x3)
        feats.append(x4.clone())
        x5 = self.features5(x4)
        feats.append(x5.clone())
        return feats


    def calLayer0(self,feats):
        sum0 = self.conv0_down(self.msblock0(feats[0]))
        # sum0 = self.non_local_0(sum0)
        return sum0
    def calLayer1(self,feats):
        sum1 = self.conv1_1_down(self.msblock1_1(feats[1])) + \
                self.conv1_2_down(self.msblock1_2(feats[2])) + \
                    self.conv1_3_down(self.msblock1_3(feats[3]))
        #sum1 = self.non_local_1(sum1)
        return sum1
    def calLayer2(self,feats):
        sum2 = self.conv2_1_down(self.msblock2_1(feats[4])) + \
            self.conv2_2_down(self.msblock2_2(feats[5]))+\
                self.conv2_3_down(self.msblock2_3(feats[6]))+\
                    self.conv2_4_down(self.msblock2_4(feats[7]))
        #sum2 = self.non_local_2(sum2)
        return sum2
    
    
    def calLayer3(self,feats):
        sum3 = self.conv3_1_down(self.msblock3_1(feats[8])) + \
            self.conv3_2_down(self.msblock3_2(feats[9])) + \
                self.conv3_3_down(self.msblock3_3(feats[10])) + \
                    self.conv3_4_down(self.msblock3_4(feats[11])) + \
                        self.conv3_5_down(self.msblock3_5(feats[12])) + \
                            self.conv3_6_down(self.msblock3_6(feats[13])) + \
                                self.conv3_7_down(self.msblock3_7(feats[14])) + \
                                    self.conv3_8_down(self.msblock3_8(feats[15])) + \
                                        self.conv3_9_down(self.msblock3_9(feats[16])) + \
                                            self.conv3_10_down(self.msblock3_10(feats[17])) + \
                                                self.conv3_11_down(self.msblock3_11(feats[18])) + \
                                                    self.conv3_12_down(self.msblock3_12(feats[19])) + \
                                                        self.conv3_13_down(self.msblock3_13(feats[20])) + \
                                                            self.conv3_14_down(self.msblock3_14(feats[21])) + \
                                                                self.conv3_15_down(self.msblock3_15(feats[22])) + \
                                                                    self.conv3_16_down(self.msblock3_16(feats[23])) + \
                                                                        self.conv3_17_down(self.msblock3_17(feats[24])) + \
                                                                            self.conv3_18_down(self.msblock3_18(feats[25])) + \
                                                                                self.conv3_19_down(self.msblock3_19(feats[26])) + \
                                                                                    self.conv3_20_down(self.msblock3_20(feats[27])) + \
                                                                                        self.conv3_21_down(self.msblock3_21(feats[28])) + \
                                                                                            self.conv3_22_down(self.msblock3_22(feats[29])) + \
                                                                                                self.conv3_23_down(self.msblock3_23(feats[30])) 
        #sum3 = self.non_local_3(sum3)
        return sum3    
    
    def calLayer4(self,feats):
        sum4 = self.conv4_1_down(self.msblock4_1(feats[31])) + \
            self.conv4_2_down(self.msblock4_2(feats[32])) + \
            self.conv4_3_down(self.msblock4_3(feats[33]))
        #sum4 = self.non_local_4(sum4)
        return sum4
    def _vgg_calLayer0(self, feats):
        sum1 = self.conv1_1_down(self.msblock1_1(feats[0]))
        return sum1
    def _vgg_calLayer1(self, feats):
        sum2 = self.conv2_1_down(self.msblock2_1(feats[1]))
        return sum2
    def _vgg_calLayer2(self, feats):
        sum3 = self.conv3_1_down(self.msblock3_1(feats[2]))
        return sum3

    def _vgg_calLayer3(self, feats):
        sum4 = self.conv4_1_down(self.msblock4_1(feats[3]))
        return sum4
    def _vgg_calLayer4(self, feats):
        sum5 = self.conv5_1_down(self.msblock5_1(feats[4]))
        return sum5

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
            calLayer2 = self.calLayer2
            calLayer1 = self.calLayer1
            calLayer0 = self.calLayer0
        elif self.backbone_name == "vgg16":
            calLayer4 = self._vgg_calLayer4
            calLayer3 = self._vgg_calLayer3
            calLayer2 = self._vgg_calLayer2
            calLayer1 = self._vgg_calLayer1
            #calLayer0 = self._vgg_calLayer0

        
        #sum0 = calLayer0(feats)
        sum1 = calLayer1(feats)
        sum2 = calLayer2(feats)
        sum3 = calLayer3(feats)
        sum4 = calLayer4(feats)

        sum4_upsamples =  F.interpolate(sum4, sum3.shape[2:], mode='bilinear',align_corners=True)
        sum3 = self.conv3_scale(torch.cat((sum3, sum4_upsamples), 1))
        sum3_upsamples = F.interpolate(sum3, sum2.shape[2:], mode='bilinear',align_corners=True)
        sum2 = self.conv2_scale(torch.cat((sum2, sum3_upsamples), 1))
        sum2_upsamples = F.interpolate(sum2, sum1.shape[2:], mode='bilinear',align_corners=True)
        sum1 = self.conv1_scale(torch.cat((sum1, sum2_upsamples), 1))
        '''
        sum1_down = self.conv0_scale(sum1)
        sum0 =F.interpolate(sum1_down, x.shape[2:], mode='bilinear',align_corners=True)
        here we get sum1 ,sum2,sum3,sum4
        '''
        x4_density = self.density_head4(sum4)
        x3_density = self.density_head3(sum3)
        x2_density = self.density_head2(sum2)
        x1_density = self.density_head1(sum1)
        # confidence prediction
        x4_confi = self.confidence_head4(sum4)
        x3_confi = self.confidence_head3(sum3)
        x2_confi = self.confidence_head2(sum2)
        x1_confi = self.confidence_head1(sum1)
        # upsample the density prediction to be the same with the input size
        x4_density = F.interpolate(x4_density, size=x.size()[2:],mode='bilinear',align_corners=True)
        x3_density = F.interpolate(x3_density, size=x.size()[2:],mode='bilinear',align_corners=True)
        x2_density = F.interpolate(x2_density, size=x.size()[2:],mode='bilinear',align_corners=True)
        x1_density = F.interpolate(x1_density, size=x.size()[2:],mode='bilinear',align_corners=True)
        # upsample the confidence prediction to be the same with the input size
        x4_confi_upsample = F.interpolate(x4_confi, size=x.size()[2:],mode='bilinear',align_corners=True)
        x3_confi_upsample = F.interpolate(x3_confi, size=x.size()[2:],mode='bilinear',align_corners=True)
        x2_confi_upsample = F.interpolate(x2_confi, size=x.size()[2:],mode='bilinear',align_corners=True)
        x1_confi_upsample = F.interpolate(x1_confi, size=x.size()[2:],mode='bilinear',align_corners=True)
 
        confidence_map = torch.cat([x4_confi_upsample,
                                    x3_confi_upsample, x2_confi_upsample, x1_confi_upsample], 1)
        # use softmax to normalize
        confidence_map = torch.nn.functional.softmax(confidence_map, 1)

        density_map = torch.cat([x4_density, x3_density, x2_density, x1_density], 1)
        # soft selection
        density_map *= confidence_map
        density = torch.sum(density_map, 1, keepdim=True)
        return density
        #print(density_map[:,0,:,:].shape)
        #return density_map[:,1,:,:].unsqueeze(1)     
        

if __name__ == '__main__':
    device = 'cuda:0'
    input_demo = torch.rand((2, 3, 224, 224)).to(device)
    target_demo = torch.rand((2, 1, 224, 224)).to(device)

    model = MCCCN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.95, weight_decay=5e-4)

    output_demo = model(input_demo)
    print(output_demo.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #self.layer4 = nn.Sequential(
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #    )        
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #self.layer8 = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #    ) 
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        #self.layer12 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #    ) 
        
    def forward(self, x):
        #Conv1
        # x = self.layer1(x)
        # x = self.layer2(x) + x
        # x = self.layer3(x) + x

        # 修改Conv1的连接方式
        output_layer1 = self.layer1(x)
        output_layer2 = self.layer2(output_layer1)
        output_layer2_add = output_layer2 + output_layer1
        output_layer3 = self.layer3(output_layer2_add) + output_layer2 + output_layer1

        #Conv2
        # x = self.layer5(x)
        # x = self.layer6(x) + x
        # x = self.layer7(x) + x

        # 修改Conv2的连接方式
        output_layer5 = self.layer5(output_layer3)
        output_layer6 = self.layer6(output_layer5)
        output_layer6_add = output_layer6 + output_layer5
        output_layer7 = self.layer7(output_layer6_add) + output_layer6 + output_layer5

        #Conv3
        # x = self.layer9(x)
        # x = self.layer10(x) + x
        # x = self.layer11(x) + x

        # 修改Conv3的连接方式
        output_layer9 = self.layer9(output_layer7)
        output_layer10 = self.layer10(output_layer9)
        output_layer10_add = output_layer10 + output_layer9
        output_layer11 = self.layer11(output_layer10_add) + output_layer10 + output_layer9

        return output_layer11

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        #self.layer15 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #    ) 
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #self.layer19 = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #    )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #self.layer23 = nn.Sequential(
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #    ) 
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self,x):        
        # #Deconv3
        # x = self.layer13(x) + x
        # x = self.layer14(x) + x
        # x = self.layer16(x)

        # 修改Deconv3的连接方式
        output_layer13 = self.layer13(x)
        output_layer13_add = output_layer13 + x
        output_layer14 = self.layer14(output_layer13_add)+output_layer13_add
        output_layer16 = self.layer16(output_layer14)

        # #Deconv2
        # x = self.layer17(x) + x
        # x = self.layer18(x) + x
        # x = self.layer20(x)

        # 修改Deconv2的连接方式
        output_layer17 = self.layer17(output_layer16)
        output_layer17_add = output_layer17 + output_layer16
        output_layer18 = self.layer18(output_layer17_add)+output_layer17_add
        output_layer20 = self.layer20(output_layer18)

        # #Deconv1
        # x = self.layer21(x) + x
        # x = self.layer22(x) + x
        # x = self.layer24(x)

        # 修改Conv1的连接方式
        output_layer21 = self.layer21(output_layer20)
        output_layer21_add = output_layer21 + output_layer20
        output_layer22 = self.layer22(output_layer21_add)+output_layer21_add
        output_layer24 = self.layer24(output_layer22)
        return output_layer24


class DIP(nn.Module):
    def __init__(self):
        super(DIP, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return output

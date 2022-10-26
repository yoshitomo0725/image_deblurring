from torch import nn
import torch
from torch.nn.functional import relu
from torchinfo import summary

# SRCNN
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        return x

# U-Net
class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64) # cannnel数(3 -> 64 -> 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) # cannnel数(1->2)
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1) # skip connection
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return x
    
    
class Unet(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet,self).__init__()

        # Encoder
        self.copu1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        for i in range(2, 6):
            self.add_module('copu%d'%i,
                nn.Sequential(
                    nn.Conv2d(48, 48, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )

        # Decoder
        self.coasa1 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )

        self.coasa2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )

        for i in range(3,6):
            self.add_module('coasa%d'%i,
                nn.Sequential(
                    nn.Conv2d(144, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
                )
            )

        self.coli = nn.Sequential(
            nn.Conv2d(96+in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self,x):
        x1 = self.copu1(x)
        x2 = self.copu2(x1)
        x3 = self.copu3(x2)
        x4 = self.copu4(x3)
        x5 = self.copu5(x4)

        y1 = self.coasa1(x5)
        z1 = torch.cat([x4, y1], dim = 1)
        y2 = self.coasa2(z1)
        
        z2 = torch.cat([x3, y2], dim = 1)
        y3 = self.coasa3(z2)

        z3 = torch.cat([x2, y3], dim = 1)
        y4 = self.coasa4(z3)
        
        z4 = torch.cat([x1, y4], dim = 1)
        y5 = self.coasa5(z4)
        z5 = torch.cat([x, y5], dim = 1)
        
        out = self.coli(z5)

        return out

class Win5RB(nn.Module):
    def __init__(self,cn=3):
        super(Win5RB,self).__init__()

        inc = cn
        for i in range(1,5):
            self.add_module('coba%d'%i,
                nn.Sequential(
                    nn.Conv2d(inc,64,7,stride=1,padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
            )
            inc = 64

        self.add_module('coba5',nn.Sequential(
                nn.Conv2d(64,cn,7,stride=1,padding=3),
                nn.BatchNorm2d(cn)
            )
        )

        for l in self.modules(): # 重みの初期値
            if(type(l)==nn.Conv2d):
                nn.init.kaiming_normal_(l.weight.data)
                l.bias.data.zero_()
            elif(type(l)==nn.BatchNorm2d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()

    def forward(self,x):
        z = self.coba1(x)
        z = self.coba2(z)
        z = self.coba3(z)
        z = self.coba4(z)
        return self.coba5(z) + x
    
#model = UNet_2D()
#model = SRCNN()
#model = Unet()
#model_info = summary(model, input_size=(3, 572, 572)) #input_size=(3, 572, 572)) #summary(model, input_size=(3, 266, 266))

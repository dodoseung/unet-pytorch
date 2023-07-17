import torch
import torch.nn.functional as F
import torch.nn as nn

# U-Net
class UNet(nn.Module):
    def __init__(self, input_dim=3, num_class=35):
        super(UNet, self).__init__()
        # Initiation
        self.input_dim = input_dim
        self.num_class = num_class
        
        # Batch norm
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        self.batch_norm512 = nn.BatchNorm2d(512)
        self.batch_norm1024 = nn.BatchNorm2d(1024)
        
        # Encoder (Contracting path)
        self.maxpool = nn.MaxPool2d((2, 2))
        
        self.enc_conv1_1 = nn.Conv2d(self.input_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.enc_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.enc_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.enc_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        self.enc_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.enc_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.enc_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.enc_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.enc_conv5_1 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.enc_conv5_2 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        # Decoder (Expanding path)
        self.upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv1_1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv1_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv2_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv2_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv3_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv4_1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv4_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.dec_conv5 = nn.Conv2d(64, self.num_class, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
    def forward(self, x):
        input_img_size = (x.size(2), x.size(3))
        
        # Encoding
        x = F.relu(self.batch_norm64(self.enc_conv1_1(x)))
        x = F.relu(self.batch_norm64(self.enc_conv1_2(x)))
        res1 = x

        x = self.maxpool(x)
        x = F.relu(self.batch_norm128(self.enc_conv2_1(x)))
        x = F.relu(self.batch_norm128(self.enc_conv2_2(x)))
        res2 = x

        x = self.maxpool(x)
        x = F.relu(self.batch_norm256(self.enc_conv3_1(x)))
        x = F.relu(self.batch_norm256(self.enc_conv3_2(x)))
        res3 = x

        x = self.maxpool(x)
        x = F.relu(self.batch_norm512(self.enc_conv4_1(x)))
        x = F.relu(self.batch_norm512(self.enc_conv4_2(x)))
        res4 = x

        x = self.maxpool(x)
        x = F.relu(self.batch_norm1024(self.enc_conv5_1(x)))
        x = F.relu(self.batch_norm1024(self.enc_conv5_2(x)))

        # Decoding
        x = self.upsample1(x)
        y = res4[:, :, res4.size(2)//2 - x.size(2)//2 : res4.size(2)//2 + x.size(2)//2, res4.size(3)//2 - x.size(3)//2 : res4.size(3)//2 + x.size(3)//2]
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.batch_norm512(self.dec_conv1_1(x)))
        x = F.relu(self.batch_norm512(self.dec_conv1_2(x)))
        
        x = self.upsample2(x)
        y = res3[:, :, res3.size(2)//2 - x.size(2)//2 : res3.size(2)//2 + x.size(2)//2, res3.size(3)//2 - x.size(3)//2 : res3.size(3)//2 + x.size(3)//2]
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.batch_norm256(self.dec_conv2_1(x)))
        x = F.relu(self.batch_norm256(self.dec_conv2_2(x)))
        
        x = self.upsample3(x)
        y = res2[:, :, res2.size(2)//2 - x.size(2)//2 : res2.size(2)//2 + x.size(2)//2, res2.size(3)//2 - x.size(3)//2 : res2.size(3)//2 + x.size(3)//2]
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.batch_norm128(self.dec_conv3_1(x)))
        x = F.relu(self.batch_norm128(self.dec_conv3_2(x)))
        
        x = self.upsample4(x)
        y = res1[:, :, res1.size(2)//2 - x.size(2)//2 : res1.size(2)//2 + x.size(2)//2, res1.size(3)//2 - x.size(3)//2 : res1.size(3)//2 + x.size(3)//2]
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.batch_norm64(self.dec_conv4_1(x)))
        x = F.relu(self.batch_norm64(self.dec_conv4_2(x)))

        x = self.dec_conv5(x)
        
        # Matching sizes of input and output
        x = F.interpolate(x, size=(input_img_size[0], input_img_size[1]), mode='bicubic', align_corners=False)
        
        return x
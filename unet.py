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
        
        # Encoder (Contracting path)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)
        
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
        self.upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(1, 1))
        self.dec_conv1_1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv1_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(1, 1))
        self.dec_conv2_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv2_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(1, 1))
        self.dec_conv3_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 1))
        self.dec_conv4_1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dec_conv4_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        self.dec_conv5 = nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
    def forward(self, x):
        # Encoding
        x = F.relu(self.enc_conv1_2(F.relu(self.enc_conv1_1(x))))
        res1 = x
        x = F.relu(self.enc_conv2_2(F.relu(self.enc_conv2_1(self.maxpool(x)))))
        res2 = x
        x = F.relu(self.enc_conv3_2(F.relu(self.enc_conv3_1(self.maxpool(x)))))
        res3 = x
        x = F.relu(self.enc_conv4_2(F.relu(self.enc_conv4_1(self.maxpool(x)))))
        res4 = x
        x = F.relu(self.enc_conv5_2(F.relu(self.enc_conv5_1(self.maxpool(x)))))

        # Decoding

        return 0
import torch
import torch.nn as nn
import torch.optim as optim

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
                                nn.Conv2d(kernel_size=3, 
                                          in_channels=1, 
                                          out_channels=32, 
                                          stride=1, padding=0),
                                nn.LeakyReLU()
                            )
        self.conv2 = nn.Sequential(
                                nn.Conv2d(kernel_size=3, 
                                          in_channels=32, 
                                          out_channels=64, 
                                          stride=1, padding=0),
                                nn.LeakyReLU()
                            )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
                                nn.Conv2d(kernel_size=3, 
                                            in_channels=64, 
                                            out_channels=128, 
                                            stride=1, padding=0),
                                nn.LeakyReLU()
                            )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
                                nn.Conv2d(kernel_size=3, 
                                            in_channels=128, 
                                            out_channels=256, 
                                            stride=1, padding=0),
                                nn.LeakyReLU()
                            )
        
        self.conv5 = nn.Sequential(
                                nn.Conv2d(kernel_size=3, 
                                            in_channels=256, 
                                            out_channels=128, 
                                            stride=1, padding=0),
                                nn.LeakyReLU()
                            )

        self.deconv4 = nn.Sequential(
                                nn.ConvTranspose2d(kernel_size=4, 
                                            in_channels=128, 
                                            out_channels=128, 
                                            stride=1, padding=0, output_padding=0),
                                nn.LeakyReLU()
                            )
        
        self.deconv5 = nn.Sequential(
                                nn.ConvTranspose2d(kernel_size=3, 
                                            in_channels=128, 
                                            out_channels=128, 
                                            stride=2, padding=0, output_padding=1),
                                nn.LeakyReLU()
                            )
                
        self.conv4_prime = nn.Sequential(
                                nn.Conv2d(kernel_size=3, 
                                            in_channels=256, 
                                            out_channels=128, 
                                            stride=1, padding=0),
                                nn.LeakyReLU()
                            )
        
        self.deconv6 = nn.Sequential(
                                nn.ConvTranspose2d(kernel_size=4, 
                                            in_channels=128, 
                                            out_channels=64, 
                                            stride=1, padding=0, output_padding=0),
                                nn.LeakyReLU()
                                    )
        
        self.deconv7 = nn.Sequential(
                                nn.ConvTranspose2d(kernel_size=3, 
                                            in_channels=64, 
                                            out_channels=64, 
                                            stride=2, padding=0, output_padding=1),
                                nn.LeakyReLU()
                            )      
          
        self.conv3_prime = nn.Sequential(
                                    nn.Conv2d(kernel_size=3, 
                                                in_channels=128, 
                                                out_channels=64, 
                                                stride=1, padding=2),
                                    nn.LeakyReLU()
                                )
        self.conv2_prime = nn.Sequential(
                                        nn.Conv2d(kernel_size=3, 
                                                    in_channels=64, 
                                                    out_channels=32, 
                                                    stride=1, padding=1),
                                        nn.LeakyReLU()
                                    )
        self.conv1_prime = nn.Sequential(
                                        nn.Conv2d(kernel_size=3, 
                                                    in_channels=32, 
                                                    out_channels=5, 
                                                    stride=1, padding=2)
                                    )
                
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.pool1(x2)
        x4 = self.conv3(x3)
        x5 = self.pool2(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x4_prime = self.deconv4(x7)
        x4_prime = self.deconv5(x4_prime)
        x4_prime = torch.concatenate((x4, x4_prime), dim=1)
        x4_prime = self.conv4_prime(x4_prime)
        x2_prime = self.deconv6(x4_prime)
        x2_prime = self.deconv7(x2_prime)
        x2_prime = torch.concatenate((x2, x2_prime), dim=1)
        x1_prime = self.conv3_prime(x2_prime)
        x1_prime = self.conv2_prime(x1_prime)
        x1_prime = self.conv1_prime(x1_prime)
        outs = x1_prime
        return x2_prime
        
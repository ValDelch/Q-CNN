import torch
import torch.nn as nn
import torch.nn.functional as F
from acsconv.operators import ACSConv, SoftACSConv
from building_blocks import QConv, QBatchNorm, QMaxPool2D, QChannelAttention, QSpatialAttention, QLinear, ConvertToQuaternions, QGlobalMaxPool2D

class Vanilla_3DCNN(nn.Module):
    def __init__(self, n_classes=2, channel_factor=1.):
        super().__init__()
        
        layers = []
        
        # Block 1
        layers.append(nn.Conv3d(in_channels=1, out_channels=int(16*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(16*channel_factor)))
        layers.append(nn.Conv3d(in_channels=int(16*channel_factor), out_channels=int(32*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(32*channel_factor)))
        layers.append(nn.Dropout(0.2))
        
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        # Block 2
        layers.append(nn.Conv3d(in_channels=int(32*channel_factor), out_channels=int(64*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(64*channel_factor)))
        layers.append(nn.Conv3d(in_channels=int(64*channel_factor), out_channels=int(128*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(128*channel_factor)))
        layers.append(nn.Dropout(0.2))
        
        self.encoder = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(128*channel_factor), out_features=int(64*channel_factor)),
            nn.Dropout(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(in_features=int(64*channel_factor), out_features=n_classes),
            nn.Dropout(p=0.2, inplace=False)
        )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        
        # Get the features
        out = self.encoder(x)
        out = F.max_pool3d(out, kernel_size=out.size()[2:]).squeeze()
        
        return self.classifier(out)
    
class Vanilla_2DCNN(nn.Module):
    def __init__(self, n_classes=2, channel_factor=1., in_channels=64):
        super().__init__()
        
        layers = []
        
        # Block 1
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=int(16*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(num_features=int(16*channel_factor)))
        layers.append(nn.Conv2d(in_channels=int(16*channel_factor), out_channels=int(32*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(num_features=int(32*channel_factor)))
        layers.append(nn.Dropout(0.2))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 2
        layers.append(nn.Conv2d(in_channels=int(32*channel_factor), out_channels=int(64*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(num_features=int(64*channel_factor)))
        layers.append(nn.Conv2d(in_channels=int(64*channel_factor), out_channels=int(128*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(num_features=int(128*channel_factor)))
        layers.append(nn.Dropout(0.2))
        
        self.encoder = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(128*channel_factor), out_features=int(64*channel_factor)),
            nn.Dropout(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(in_features=int(64*channel_factor), out_features=n_classes),
            nn.Dropout(p=0.2, inplace=False)
        )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        
        # Get the features
        out = self.encoder(x)
        out = F.max_pool2d(out, kernel_size=out.size()[2:]).squeeze()
        
        return self.classifier(out)
    
class QNet(nn.Module):
    def __init__(self, width=128, n_classes=2, upsampling_factor=1., channel_factor=1., isotrope=False):
        super().__init__()
        
        layers = []
        
        # Block 1
        layers.append(ConvertToQuaternions(width=width, upsample_factor=upsampling_factor, isotrope=isotrope))
        layers.append(nn.ReLU())
        
        layers.append(QConv(k=3, C_in=int(upsampling_factor*width), C_out=int(16*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(16*channel_factor)))
        layers.append(QConv(k=3, C_in=int(16*channel_factor), C_out=int(32*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(32*channel_factor)))
        layers.append(nn.Dropout(p=0.2))

        layers.append(QMaxPool2D(kernel_size=2, stride=2))
        
        # Block 2
        layers.append(QConv(k=3, C_in=int(32*channel_factor), C_out=int(64*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(64*channel_factor)))
        layers.append(QConv(k=3, C_in=int(64*channel_factor), C_out=int(128*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(128*channel_factor)))
        layers.append(nn.Dropout(p=0.2))
        
        layers.append(QGlobalMaxPool2D())
        
        self.encoder = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            QLinear(C_in=int(128*channel_factor), C_out=int(64*channel_factor)),
            nn.Dropout(p=0.2, inplace=False),
            nn.ReLU(),
            QLinear(C_in=int(64*channel_factor), C_out=n_classes),
            nn.Dropout(p=0.2, inplace=False)
        )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        
        # Get the quaternionic features
        out = self.encoder(x) # (b,C_out,x,x,4)
        
        # Dense part
        out = self.classifier(out) # (b, n_classes, 4)
        #out = torch.sum(torch.square(out), dim=2, keepdim=True)
        
        return out[:,:,0]
    
class ACS_CNN(nn.Module):
    def __init__(self, n_classes=2, channel_factor=1.7):
        super().__init__()
        
        layers = []
        
        # Block 1
        layers.append(ACSConv(in_channels=1, out_channels=int(16*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(16*channel_factor)))
        layers.append(ACSConv(in_channels=int(16*channel_factor), out_channels=int(32*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(32*channel_factor)))
        layers.append(nn.Dropout(0.2))

        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        # Block 2
        layers.append(ACSConv(in_channels=int(32*channel_factor), out_channels=int(64*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(64*channel_factor)))
        layers.append(ACSConv(in_channels=int(64*channel_factor), out_channels=int(128*channel_factor), kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm3d(num_features=int(128*channel_factor)))
        layers.append(nn.Dropout(0.2))
        
        self.encoder = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(128*channel_factor), out_features=int(64*channel_factor)),
            nn.Dropout(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(in_features=int(64*channel_factor), out_features=n_classes),
            nn.Dropout(p=0.2, inplace=False)
        )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        
        # Get the features
        out = self.encoder(x)
        out = F.max_pool3d(out, kernel_size=out.size()[2:]).squeeze()
        
        return self.classifier(out)
    
class QNetSpatial(nn.Module):
    def __init__(self, width=128, n_classes=2, upsampling_factor=1., channel_factor=1.):
        super().__init__()
        
        layers = []
        
        # Block 1
        layers.append(ConvertToQuaternions(width=width, upsample_factor=upsampling_factor, isotrope=False))
        layers.append(nn.ReLU())
        
        layers.append(QConv(k=3, C_in=int(upsampling_factor*width), C_out=int(16*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(16*channel_factor)))
        layers.append(QConv(k=3, C_in=int(16*channel_factor), C_out=int(32*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(32*channel_factor)))
        layers.append(nn.Dropout(p=0.2))

        layers.append(QChannelAttention(C_in=int(32*channel_factor)))
        layers.append(QMaxPool2D(kernel_size=2, stride=2))
        
        # Block 2
        layers.append(QConv(k=3, C_in=int(32*channel_factor), C_out=int(64*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(64*channel_factor)))
        layers.append(QConv(k=3, C_in=int(64*channel_factor), C_out=int(128*channel_factor), padding=1))
        layers.append(nn.ReLU())
        layers.append(QBatchNorm(C_in=int(128*channel_factor)))
        layers.append(nn.Dropout(p=0.2))
        
        layers.append(QChannelAttention(C_in=int(128*channel_factor)))
        layers.append(QGlobalMaxPool2D())
        
        self.encoder = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            QLinear(C_in=int(128*channel_factor), C_out=int(64*channel_factor)),
            nn.Dropout(p=0.2, inplace=False),
            nn.ReLU(),
            QLinear(C_in=int(64*channel_factor), C_out=n_classes),
            nn.Dropout(p=0.2, inplace=False)
        )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        
        # Get the quaternionic features
        out = self.encoder(x) # (b,C_out,x,x,4)
        
        # Dense part
        out = self.classifier(out) # (b, n_classes, 4)
        #out = torch.sqrt(torch.sum(torch.square(out), dim=2, keepdim=True))
        
        return out[:,:,0]

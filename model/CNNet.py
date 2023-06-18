import torch
import torch.nn as nn
import numpy as np

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), # 32,128,128     # first value is "pages"
            nn.MaxPool2d(2),                # 32,64,64
            nn.Conv2d(32,64,3,padding=1),   # 64,64,64
            nn.MaxPool2d(2),                # 64,32,32
            nn.Conv2d(64,128,3,padding=1),  # 128,32,32
            nn.MaxPool2d(2),                # 128,16,16
            nn.Conv2d(128,256,3,padding=1), # 256,16,16
            nn.MaxPool2d(2)                 # 256,8,8
        )
        self.fc = nn.Sequential(
            nn.Linear(16384,4)  # 4 = no,very mild, mild, moderate
        )

        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(out.size(0),-1)    # flattten [255*8*8]
        out = self.fc(out)
        return out

class CNNet_convtrans(nn.Module):
    def __init__(self):
        super(CNNet_convtrans,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=4,stride=2,padding=1), # 1,32,64,64    # first value is "pages"
            nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),                 # 64,32,32
            
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.Conv2d(64,64,kernel_size=4,stride=2,padding=1),           # 128,16,16
            
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.Conv2d(128,128,kernel_size=4,stride=2,padding=1),                 # 256,8,8
       
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1), 
        )
        self.fc = nn.Sequential(
            nn.Linear(16384,4)  # 4 = no,very mild, mild, moderate
        )

        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(out.size(0),-1)    # flattten [255*8*8]
        out = self.fc(out)
        return out


model = CNNet()

in_data = torch.tensor(torch.ones(128,128))
in_data = in_data.unsqueeze(0).unsqueeze(0)   # [1,1,128,128]
print(in_data.shape)


result = model(in_data)


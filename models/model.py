import torch.nn as nn
import torchvision

class R2plus1d_18(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super().__init__()
        self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x


        
         
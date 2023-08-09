import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class TeS_model(nn.Module):
    def __init__(self, num_classes:int, pretrained=None):
        super().__init__()

        self.model = models.resnet50()
        if pretrained:
            if os.path.isfile(pretrained):
                print("=> loading checkpoint '{}'".format(pretrained))
                checkpoint = torch.load(pretrained, map_location="cpu")
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder_q.') and not k.startswith('module.encoder_q.fc'):
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    del state_dict[k]
                self.model.load_state_dict(state_dict, strict=False)
            else:
                print("=> no checkpoint found at '{}'".format(pretrained))
                sys.exit(-1)
        else:
            print("=>no checkpoint, please use --pretrained to load it")
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512)
            )

        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x_proj = self.projection_head(x)
        x_proj = F.normalize(x_proj, dim=1)
        x = self.fc(x)
        return {'x': x, 'x_proj': x_proj}
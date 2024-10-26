import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class baseMLP(nn.Module):
    def __init__(self, in_dim, feature_dim=128):
        super(baseMLP, self).__init__()
        # encoder
        self.f = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )
        # projection head
        self.g = nn.Sequential(
            # nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        embedding = self.f(x)
        projection = self.g(embedding)

        # L2 normalization by default
        return F.normalize(embedding, dim=-1), F.normalize(projection, dim=-1)  # dim=-1 means along the last dimension
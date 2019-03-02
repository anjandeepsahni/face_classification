import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channel_size, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=channel_size, out_channels=channel_size,
                                            kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(num_features=channel_size),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=channel_size, out_channels=channel_size,
                                            kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(num_features=channel_size)
                                    )
        self.non_linear = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        out = self.block(out)
        out = self.non_linear(out + x)
        return out

class FaceClassifier(nn.Module):
    def __init__(self, num_feat, hidden_sizes, num_classes, feat_dim=10):
        super(FaceClassifier, self).__init__()
        self.hidden_sizes = [num_feat] + hidden_sizes + [num_classes]
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx],
                                        out_channels=self.hidden_sizes[idx+1],
                                        kernel_size=2, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(ResBlock(channel_size=channel_size))
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(in_features=self.hidden_sizes[-2], out_features=self.hidden_sizes[-1], bias=False)

        # Creating embedding for face verification task.
        self.linear_closs = nn.Linear(in_features=self.hidden_sizes[-2], out_features=feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self,x):
        out = x
        out = self.layers(x)
        # Pooling before fully connected layer to reduce params.
        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=1)
        out = out.reshape(out.shape[0], out.shape[1])

        # Fully Connected.
        label_out = self.linear_label(out)
        label_out = label_out/torch.norm(self.linear_label.weight, dim=1)   # Normalize to keep between [0,1] for embedding.

        # Embedding.
        closs_out = self.linear_closs(out)
        closs_out = self.relu_closs(closs_out)

        return closs_out, label_out

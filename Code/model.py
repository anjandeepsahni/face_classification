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

class InvertedResidualBottleneck(nn.Module):
    def __init__(self, insize, outsize, stride, expfact):
        super(InvertedResidualBottleneck, self).__init__()
        # Check if we need to use residual connection during forward pass.
        self.use_res_conn = (stride == 1 and insize == outsize)
        self.block = nn.Sequential(
            # Pointwise Expansion.
            nn.Conv2d(in_channels=insize, out_channels=(insize*expfact),
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=(insize*expfact)),
            nn.ReLU6(inplace=True),
            # Depthwise Convolution.
            nn.Conv2d(in_channels=(insize*expfact), out_channels=(insize*expfact),
                                    kernel_size=3, stride=stride, padding=1,
                                    groups=(insize*expfact), bias=False),
            nn.BatchNorm2d(num_features=(insize*expfact)),
            nn.ReLU6(inplace=True),
            # Pointwise Projection.
            nn.Conv2d(in_channels=(insize*expfact), out_channels=outsize,
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=outsize),
        )

    def forward(self, x):
        if self.use_res_conn:
            return (x + self.block(x))
        else:
            return self.block(x)

class FaceClassifier2(nn.Module):
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

class FaceClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim=10):
        super(FaceClassifier, self).__init__()
        self.infeat = 3 # Num of input channels.
        self.conv_channels = [16, 32]
        self.res_channels = [16, 32, 64, 128]
        self.res_expfact = [1, 6, 6, 6]
        self.res_repeat = [1, 3, 5, 7]
        self.res_stride = [1, 2, 2, 2]
        #self.conv_channels = [16, 32]
        #self.res_channels = [16, 24, 32, 64, 96, 160, 320]
        #self.res_channels = [16, 24, 32, 64, 96, 160, 320]
        #self.res_expfact = [1, 6, 6, 6, 6, 6, 6]
        #self.res_repeat = [1, 2, 3, 4, 3, 3, 1]
        #self.res_stride = [1, 2, 1, 2, 1, 2, 1]
        self.layers = []
        # Prepare the model.
        insize = self.infeat
        for outsize in self.conv_channels:
            self.layers += [nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=3, stride=1, bias=False)]
            self.layers += [nn.BatchNorm2d(num_features=outsize)]
            self.layers += [nn.ReLU6(inplace=True)]
            insize = outsize
        for idx, outsize in enumerate(self.res_channels):
            for n in range(self.res_repeat[idx]):
                if n == 0:
                    self.layers += [InvertedResidualBottleneck(insize=insize, outsize=outsize, stride=self.res_stride[idx],
                                                                expfact=self.res_expfact[idx])]
                    insize = outsize
                else:
                    self.layers += [InvertedResidualBottleneck(insize=insize, outsize=outsize,
                                                                stride=1, expfact=self.res_expfact[idx])]
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(in_features=self.res_channels[-1], out_features=num_classes, bias=False)

        # Creating embedding for face verification task.
        self.linear_closs = nn.Linear(in_features=self.res_channels[-1], out_features=feat_dim, bias=False)
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

class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

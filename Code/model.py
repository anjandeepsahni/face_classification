import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# For debugging sequential.
class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

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

class ResBottleneck(nn.Module):
    def __init__(self, insize, outsize, stride, expfact, downsample):
        super(ResBottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=outsize),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outsize, out_channels=outsize, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=outsize),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outsize, out_channels=(outsize*expfact), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=(outsize*expfact)),
            )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class InvResBottleneck(nn.Module):
    def __init__(self, insize, outsize, stride, expfact):
        super(InvResBottleneck, self).__init__()
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
            out = x + self.block(x)
        else:
            out = self.block(x)
        return out

# Resnet50
class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        # Initial conv layer and maxpool.
        # Kernel size changed from 7->6 and stride from 2->1
        # due to input image size difference (224->32).
        self.layers = [
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=6, bias=False),
                        nn.BatchNorm2d(num_features=64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1)
                    ]
        insize = 64
        self.res_channels = [64, 128, 256, 512]
        self.res_expfact = [4, 4, 4, 4]
        self.res_repeat = [3, 4, 6, 3]
        self.res_stride = [1, 2, 2, 2]

        for idx, outsize in enumerate(self.res_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=insize, out_channels=(outsize * self.res_expfact[idx]),
                            kernel_size=1, stride=self.res_stride[idx], bias=False),
                nn.BatchNorm2d(num_features=(outsize * self.res_expfact[idx]))
            )
            for n in range(self.res_repeat[idx]):
                if n == 0:
                    self.layers += [ResBottleneck(insize=insize, outsize=outsize,
                                    stride=self.res_stride[idx], expfact=self.res_expfact[idx], downsample=downsample)]
                else:
                    self.layers += [ResBottleneck(insize=insize, outsize=outsize,
                                    stride=1, expfact=self.res_expfact[idx], downsample=None)]
                insize = (outsize * self.res_expfact[idx])
        self.layers = nn.Sequential(*self.layers)
        # Fully connected.
        self.linear_label = nn.Linear(in_features=(self.res_channels[-1]*self.res_expfact[-1]), out_features=num_classes, bias=False)

    def forward(self, x):
        out = x
        out = self.layers(x)
        # Pooling before fully connected layer to reduce params.
        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=1)
        out = out.reshape(out.shape[0], out.shape[1])
        # Fully Connected.
        label_out = self.linear_label(out)
        label_out = label_out/torch.norm(self.linear_label.weight, dim=1)   # Normalize to keep between [0,1] for embedding.
        # Return flattened final conv layer for verification.
        # Return output of linear layer for classification.
        return out, label_out

class MobileNetV2_v1(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_v1, self).__init__()
        self.infeat = 3 # Num of input channels.
        self.conv_channels = [16, 32]
        self.res_channels = [16, 32, 64, 128]
        self.res_expfact = [1, 6, 6, 6]
        self.res_repeat = [1, 3, 5, 7]
        self.res_stride = [1, 2, 2, 2]
        self.layers = []
        # Prepare the model.
        insize = self.infeat
        for outsize in self.conv_channels:
            self.layers += [nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=3, stride=1, bias=False)]
            self.layers += [nn.BatchNorm2d(num_features=outsize), nn.ReLU6(inplace=True)]
            insize = outsize
        res_skip_proj = False
        for idx, outsize in enumerate(self.res_channels):
            for n in range(self.res_repeat[idx]):
                if n == 0:
                    self.layers += [InvResBottleneck(insize=insize, outsize=outsize, stride=self.res_stride[idx],
                                                                expfact=self.res_expfact[idx])]
                    insize = outsize
                else:
                    self.layers += [InvResBottleneck(insize=insize, outsize=outsize,
                                                                stride=1, expfact=self.res_expfact[idx])]
        self.layers = nn.Sequential(*self.layers)
        # Final label layer.
        self.linear_label = nn.Linear(in_features=self.res_channels[-1], out_features=num_classes, bias=False)

    def forward(self,x):
        out = self.layers(x)
        # Pooling before fully connected layer to reduce params.
        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=1)
        out = out.reshape(out.shape[0], out.shape[1])
        # Fully Connected - Labels.
        label_out = self.linear_label(out)
        label_out = label_out/torch.norm(self.linear_label.weight, dim=1)   # Normalize to keep between [0,1] for embedding.
        # Return flattened final conv layer for verification.
        # Return output of linear layer for classification.
        return out, label_out

class MobileNetV2_v2(nn.Module):
    def __init__(self, num_classes, feat_dim=256):
        super(MobileNetV2_v2, self).__init__()
        self.infeat = 3     # Num of input channels.
        self.feat_dim = feat_dim
        self.conv_channels = [16, 32]
        self.res_channels = [16, 32, 64, 128]
        self.res_expfact = [1, 6, 6, 6]
        self.res_repeat = [1, 3, 4, 7]
        self.res_stride = [1, 2, 1, 2]
        self.layers = []
        # Prepare the model.
        insize = self.infeat
        for outsize in self.conv_channels:
            self.layers += [nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=3, stride=1, bias=False)]
            self.layers += [nn.BatchNorm2d(num_features=outsize), nn.ReLU6(inplace=True)]
            insize = outsize
        res_skip_proj = False
        for idx, outsize in enumerate(self.res_channels):
            for n in range(self.res_repeat[idx]):
                if n == 0:
                    self.layers += [InvResBottleneck(insize=insize, outsize=outsize, stride=self.res_stride[idx],
                                                                expfact=self.res_expfact[idx])]
                    insize = outsize
                else:
                    self.layers += [InvResBottleneck(insize=insize, outsize=outsize,
                                                                stride=1, expfact=self.res_expfact[idx])]
        self.layers = nn.Sequential(*self.layers)

        # Embedding layer.
        self.embed = nn.Linear(in_features=self.res_channels[-1], out_features=feat_dim, bias=False)
        self.embed_relu = nn.ReLU(inplace=True)

        # Final label layer.
        self.dropout_layer = nn.Dropout2d(p=0.3)
        self.label = nn.Linear(in_features=self.res_channels[-1], out_features=num_classes, bias=False)

    def forward(self,x):
        out = self.layers(x)
        # Pooling before fully connected layer to reduce params.
        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=1)
        out = out.reshape(out.shape[0], out.shape[1])

        # Fully Connected - Embedding.
        embed_out = self.embed(out)
        embed_out = self.embed_relu(embed_out)

        # Fully Connected - Labels.
        labels_out = self.dropout_layer(out)
        label_out = self.label(out)
        label_out = label_out/torch.norm(self.label.weight, dim=1)   # Normalize to keep between [0,1] for embedding.
        return embed_out, label_out

class MobileNetV2_v3(nn.Module):
    def __init__(self, num_classes, feat_dim=256):
        super(MobileNetV2_v3, self).__init__()
        self.infeat = 3 # Num of input channels.
        self.feat_dim = feat_dim
        self.conv_channels = [16, 32]
        self.res_channels = [16, 32, 64, 128]
        self.res_expfact = [1, 6, 6, 6]
        self.res_repeat = [1, 3, 5, 7]
        self.res_stride = [1, 2, 2, 2]
        self.layers = []
        # Prepare the model.
        insize = self.infeat
        for outsize in self.conv_channels:
            self.layers += [nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=3, stride=1, bias=False)]
            self.layers += [nn.BatchNorm2d(num_features=outsize), nn.ReLU6(inplace=True)]
            insize = outsize
        res_skip_proj = False
        for idx, outsize in enumerate(self.res_channels):
            for n in range(self.res_repeat[idx]):
                if n == 0:
                    self.layers += [InvResBottleneck(insize=insize, outsize=outsize, stride=self.res_stride[idx],
                                                                expfact=self.res_expfact[idx])]
                    insize = outsize
                else:
                    self.layers += [InvResBottleneck(insize=insize, outsize=outsize,
                                                                stride=1, expfact=self.res_expfact[idx])]
        self.layers = nn.Sequential(*self.layers)

        # Embedding layer.
        self.embed = nn.Linear(in_features=self.res_channels[-1], out_features=feat_dim, bias=False)
        self.embed_relu = nn.ReLU(inplace=True)

        # Final label layer.
        self.dropout_layer = nn.Dropout2d(p=0.3)
        self.label = nn.Linear(in_features=self.res_channels[-1], out_features=num_classes, bias=False)

    def forward(self,x):
        out = self.layers(x)
        # Pooling before fully connected layer to reduce params.
        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=1)
        out = out.reshape(out.shape[0], out.shape[1])
        # Fully Connected - Embedding.
        embed_out = self.embed(out)
        embed_out = self.embed_relu(embed_out)
        # Fully Connected - Labels.
        labels_out = self.dropout_layer(out)
        label_out = self.label(out)
        label_out = label_out/torch.norm(self.label.weight, dim=1)   # Normalize to keep between [0,1] for embedding.
        return embed_out, label_out

# Assign an alias to single model for training time.
FaceClassifier = MobileNetV2_v3

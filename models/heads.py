import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .initializer import normal_, constant_, zeros_


class CLSHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_classes):
        super(CLSHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.convs = nn.LayerList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2D(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU())
        self.head = nn.Conv2D(feat_channels, num_anchors*num_classes, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                constant_(m.weight, 1)
                zeros_(m.bias)
        prior = 0.01
        constant_(self.head.weight, 0)
        constant_(self.head.bias, -math.log((1.0 - prior) / prior))
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = F.sigmoid(self.head(x))
        x = x.transpose([0, 2, 3, 1])
        n, w, h, c = x.shape
        x = x.reshape([n, w, h, self.num_anchors, self.num_classes])
        return x.reshape([x.shape[0], -1, self.num_classes])


class REGHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_regress = num_regress
        self.convs = nn.LayerList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2D(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU())
        self.head = nn.Conv2D(feat_channels, num_anchors*num_regress, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                constant_(m.weight, 1)
                zeros_(m.bias)
        constant_(self.head.weight, 0)
        constant_(self.head.bias, 0)
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.head(x)
        x = x.transpose([0, 2, 3, 1])
        return x.reshape([x.shape[0], -1, self.num_regress])
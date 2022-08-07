import math
import paddle
import paddle.nn as nn
import numpy as np


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
                # Torch Version:
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                new_weight = paddle.normal(mean=0.0, 
                                           std=math.sqrt(2. / n),
                                           shape=m.weight.shape)
                m.weight.set_value(new_weight)
                
            elif isinstance(m, nn.BatchNorm2D):
                
                # m.weight.data.fill_(1)
                new_weight = paddle.full(shape=m.weight.shape, 
                                         dtype=m.weight.dtype, 
                                         fill_value=1)
                m.weight.set_value(new_weight)
                
                # m.bias.data.zero_()
                new_bias = paddle.zeros(shape=m.weight.shape, 
                                        dtype=m.weight.dtype)
                m.bias.set_value(new_bias)
                
        prior = 0.01
        # self.head.weight.data.fill_(0)
        new_weight = paddle.zeros(shape=self.head.weight.shape, 
                                  dtype=self.head.weight.dtype)
        self.head.weight.set_value(new_weight)
        
        # self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))
        new_bias = paddle.full(shape=self.head.bias.shape, 
                               dtype=self.head.bias.dtype, 
                               fill_value=-math.log((1.0 - prior) / prior))
        self.head.bias.set_value(new_bias)
        

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = nn.functional.sigmoid(self.head(x))
        x = x.transpose((0, 2, 3, 1))
        n, w, h, c = x.shape
        x = x.reshape((n, w, h, self.num_anchors, self.num_classes))
        return x.reshape((x.shape[0], -1, self.num_classes))


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
        
        # Torch Version:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2D):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2D):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # self.head.weight.data.fill_(0)
        # self.head.bias.data.fill_(0)
        
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                new_weight = paddle.normal(mean=0.0, 
                                           std=math.sqrt(2. / n),
                                           shape=m.weight.shape)
                m.weight.set_value(new_weight)
                
            elif isinstance(m, nn.BatchNorm2D):
                
                new_weight = paddle.full(shape=m.weight.shape, 
                                         dtype=m.weight.dtype, 
                                         fill_value=1)
                m.weight.set_value(new_weight)
                new_bias = paddle.zeros(shape=m.weight.shape, 
                                        dtype=m.weight.dtype)
                m.bias.set_value(new_bias)
                
        new_weight = paddle.zeros(shape=self.head.weight.shape, 
                                  dtype=self.head.weight.dtype)
        self.head.weight.set_value(new_weight)
        
        new_bias = paddle.zeros(shape=self.head.bias.shape, 
                                dtype=self.head.bias.dtype)
        self.head.bias.set_value(new_bias)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.head(x)
        x = x.transpose((0, 2, 3, 1))
        return x.reshape((x.shape[0], -1, self.num_regress))
    



if __name__ == "__main__":
    
    # cls_h = CLSHead(
    #     in_channels=256,
    #     feat_channels=256,
    #     num_stacked=4,      
    #     num_anchors=16,
    #     num_classes=2)
    
    # input_ = np.random.randn(*[1, 256, 224, 224]).astype("float32")
    # input_ = paddle.to_tensor(input_)
    # out = cls_h(input_)
    # print(out.shape)
    
    reg_h = REGHead(
        in_channels=256,
        feat_channels=256,
        num_stacked=4,      
        num_anchors=2,
        num_regress=5)
    
    input_ = np.random.randn(*[1, 256, 224, 224]).astype("float32")
    input_ = paddle.to_tensor(input_)
    out = reg_h(input_)
    print(out.shape)
    
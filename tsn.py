import torch
from torch import nn
import torchvision
import tf_model_zoo


class ConsensusFunction(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, x):
        self.shape = x.size()
        if self.consensus_type == 'avg':
            x = x.mean(dim=self.dim, keepdim=True)
        if self.consensus_type == 'identity':
            x = x

        return x

    def backward(self, dout):
        if self.consensus_type == 'avg':
            din = dout.expand(self.shape) / float(self.shape[self.dim])
        if self.consensus_type == 'identity':
            din = dout

        return din

class Consensus(nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(Consensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, x):

        return ConsensusFunction(self.consensus_type, self.dim)(x)

class TSN(nn.Module):

    def __init__(self, base_model='BNInception', n_class=101, consensus_type='avg', before_softmax=True, dropout=0.8, n_crop=1, modality='RGB', n_segment=3, new_length=1):
        super(TSN, self).__init__()
        self.base_model = base_model
        self.n_class = n_class
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.n_crop = n_crop
        self.modality = modality
        self.n_segment = n_segment
        self.new_length = new_length

        print('''
        Initializing TSN with base model: {}
        TSN Configuration:
            modality:           {}
            n_segment:          {}
            new_length:         {}
            consensus_type:     {}
            dropout:            {}
        '''.format(self.base_model, self.modality, self.n_segment, self.new_length, self.consensus_type, self.dropout))

        self._prepare_base_model()
        self._prepare_tsn()
        if self.modality == 'RGBDiff':
            print('Converting the imagenet model to RGBDiff init model')
            self._construct_rgbdiff_model()
            print('Done')
        if self.modality == 'Flow':
            print('Converting the imagenet model to flow init model')
            self._construct_flow_model()
            print('Done')
        self.consensus = Consensus(self.consensus_type)
        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _prepare_base_model(self):
        if 'vgg' in self.base_model or 'resnet' in self.base_model:
            self.base_model = getattr(torchvision.models, self.base_model)(True)
            self.base_model.last_layer_name = 'fc'
            return
        if self.base_model == 'BNInception':
            self.base_model = getattr(tf_model_zoo, self.base_model)()
            self.base_model.last_layer_name = 'fc'
            return
        if 'inception' in self.base_model:
            self.base_model = getattr(tf_model_zoo, self.base_model)()
            self.base_model.last_layer_name = 'classif'
            return

    def _prepare_tsn(self):
        in_features = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(in_features=in_features, out_features=self.n_class))
            self.new_linear = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_linear = nn.Linear(in_features=in_features, out_features=self.n_class)

        if self.new_linear is None:
            nn.init.normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, 0.001)
            nn.init.constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            nn.init.normal_(self.new_linear.weight, 0, 0.001)
            nn.init.constant_(self.new_linear.bias, 0)

    def _construct_rgbdiff_model(self):
        modules = list(self.base_model.modules())
        first_conv_index = list(filter(lambda i: isinstance(modules[i], nn.Conv2d), range(len(modules))))[0]
        conv_layer = modules[first_conv_index]
        container = modules[first_conv_index - 1]

        ps = [i.clone() for i in conv_layer.parameters()]
        kernel_size = ps[0].size()
        new_kernel_size=  kernel_size[ : 1] + (3 * self.new_length, ) + kernel_size[2 : ]
        new_kernel = ps[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv_layer = nn.Conv2d(in_channels=new_kernel_size[1], out_channels=conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=True if len(ps) == 2 else False)
        new_conv_layer.weight.data = new_kernel
        if len(ps) == 2:
            new_conv_layer.bias.data = ps[1].data
        layer_name = list(container.state_dict())[0][ : -7]
        setattr(container, layer_name, new_conv_layer)

    def _construct_flow_model(self):
        modules = list(self.base_model.modules())
        first_conv_index = list(filter(lambda i: isinstance(modules[i], nn.Conv2d), range(len(modules))))[0]
        conv_layer = modules[first_conv_index]
        container = modules[first_conv_index - 1]

        ps = [i.clone() for i in conv_layer.parameters()]
        kernel_size = ps[0].size()
        new_kernel_size = kernel_size[: 1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernel = ps[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv_layer = nn.Conv2d(in_channels=2 * self.new_length, out_channels=conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=True if len(ps) == 2 else False)
        new_conv_layer.weight.data = new_kernel
        if len(ps) == 2:
            new_conv_layer.bias.data = ps[1].data
        layer_name = list(container.state_dict())[0][: -7]
        setattr(container, layer_name, new_conv_layer)

    def train(self, mode=True):
        super(TSN, self).train(mode=mode)
        count = 0
        print('Freezing BatchNorm2D except the first one')
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= 2:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x):
        sample_len = (3 if self.modality == 'RGB' or self.modality == 'RGBDiff' else 2) * self.new_length
        if self.modality == 'RGBDiff':
            x = self._get_diff(x)
        x = self.base_model(x.view((-1, sample_len) + x.size()[-2 : ]))
        if self.dropout > 0:
            x = self.new_linear(x)
        if not self.before_softmax:
            x = self.softmax(x)
        x = x.view((-1, self.n_segment) + x.size()[1 : ])
        x = self.consensus(x)
        x = x.squeeze(1)

        return x

    def _get_diff(self, x):
        x = x.view((-1, self.n_segment, self.new_length + 1, 3) + x.size()[2 : ])
        new_x = x[ : , : , 1 : , : , : , : ].clone()
        for i in reversed(range(1, self.new_length + 1, 1)):
            new_x[ : , : , i - 1, : , : , : ] = x[ : , : , i , : , : , : ] - x[ : , : , i - 1 , : , : , : ]

        return new_x
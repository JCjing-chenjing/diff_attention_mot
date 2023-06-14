
import torch
import transformer
from torch import nn
from torch import Tensor
import torchvision.transforms as T
from torch.nn import functional as F
from torchvision.models import resnet50
from typing import Optional, Any, Union, Callable
from torch.nn.modules.normalization import LayerNorm


class DIFFTrack(nn.Module):
    def __init__(self, num_classes,  hidden_dim: int = 256, nheads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, d_model: int = 512, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DIFFTrack).__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv1 = nn.Conv2d(2048, hidden_dim, 1)
        self.conv2 = nn.Conv2d(2*hidden_dim, hidden_dim, 1)
        # create a default PyTorch transformer
        # self.transformer = nn.Transformer(
        #     hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # self.transformer = transformer.Transformer(
        #     hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        if custom_encoder:
            self.encoder = custom_encoder
        else:
            base_encoder = transformer.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = transformer.TransformerEncoder(base_encoder, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            base_decoder = transformer.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder =transformer.TransformerDecoder(base_decoder, num_decoder_layers, decoder_norm)

        # # prediction heads, one extra class for predicting non-empty slots
        # # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # self.linear_bbox = nn.Linear(hidden_dim, 4)

        # # output positional encodings (object queries)
        # self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # # spatial positional encodings
        # # note that in baseline DETR we use sine positional encodings
        # self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def get_feature(self, img):
        #cnn get feature
        out = self.backbone.conv1(img)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        out = self.backbone.layer1(out)
        out = self.backbone.layer2(out)
        out = self.backbone.layer3(out)
        out = self.backbone.layer4(out)

        out = self.conv1(out)
        return out

    def forward(self, pre_img, cur_img, pre_detect):
        pre_out = self.get_feature(pre_img)
        cur_out = self.get_feature(cur_img)
        #Splice two images feature
        feature = torch.cat([pre_out, cur_out])
        #Calculate the Mutual information of two images feature
        diff_feature = self.conv2(feature)
        self.encoder(diff_feature)


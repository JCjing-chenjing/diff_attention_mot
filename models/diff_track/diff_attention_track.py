
import torch
import math
import models.diff_track.transformer as transformer
from torch import nn
from torch import Tensor
import torchvision.transforms as T
from torch.nn import functional as F
from torchvision.models import resnet50
from typing import Optional, Any, Union, Callable
from torch.nn.modules.normalization import LayerNorm


class DIFFTrack(nn.Module):
    def __init__(self, args, num_classes=2,  hidden_dim: int = 256, nheads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DIFFTrack, self).__init__()

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
        self.norm_layer = nn.BatchNorm2d(hidden_dim)
        # self.normal2 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

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
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # # output positional encodings (object queries)
        # self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(64, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(64, hidden_dim // 2))

    def box2embed(self, pos, num_pos_feats=64, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        posemb = pos[..., None] / dim_t
        posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
        return posemb

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
        out = self.norm_layer(out)  
        out = self.relu(out)
        return out

    def forward(self, pre_img, cur_img, pre_boxes):
        pre_out = self.get_feature(pre_img)
        cur_out = self.get_feature(cur_img)
        #Splice two images feature
        feature = torch.cat([pre_out, cur_out], 1)  # [2,256,17,23]
        #Calculate the Mutual information of two images feature
        diff_feature = self.conv2(feature)
        diff_feature = self.norm_layer(diff_feature)  
        diff_feature = self.relu(diff_feature)

        H, W = diff_feature.shape[-2:]

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        encoder_out = self.encoder(pos + 0.1 * diff_feature.flatten(2).permute(2, 0, 1))




        query_embed = self.box2embed(pre_boxes).permute(1,0,2)
        decoder_out = self.decoder(query_embed, encoder_out)
        box_classes = self.linear_class(decoder_out).permute(1,0,2)
        pred_boxes = self.linear_bbox(decoder_out).permute(1,0,2)
        return box_classes, pred_boxes
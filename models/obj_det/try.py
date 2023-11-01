
import torch
from models.obj_det.transformer_obj import TransformerDec







if __name__ == '__main__':

    M=TransformerDec(d_model=256, output_intermediate_dec=True,num_classes=4)

    src=torch.rand((391,2,256))

    pos_embed=torch.ones((391,1,256))

    res = M(src,pos_embed)


















import torch.nn as nn
import torch.nn.functional as F




# x = torch.rand((3,3))
# y = torch.tensor([0,1,1])

#x的值
#tensor([[0.7459, 0.5881, 0.4795],
#        [0.2894, 0.0568, 0.3439],
#        [0.6124, 0.7558, 0.4308]])

#y的值
#tensor([0, 1, 1])




class General_Loss():

    def __init__(self,num_class, **kwargs):
        self.cross=nn.CrossEntropyLoss()
        self.num_class=num_class
        self.reduction='mean'  # sum mean none



    def create_loss(self,outputs,targets):
        pred_logits=outputs['pred_logits']
        pred_boxes=outputs['pred_boxes']

        device=pred_logits.device

        boxes=targets['boxes'].to(device)
        score=targets['scores'].to(device)
        score_loss=self.cross(pred_logits[0],score[0].long())

        box_loss=F.smooth_l1_loss(boxes, pred_boxes, reduction=self.reduction)



        loss_result=score_loss+box_loss
        return loss_result


















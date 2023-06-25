


import torch.nn as nn





# x = torch.rand((3,3))
# y = torch.tensor([0,1,1])

#x的值
#tensor([[0.7459, 0.5881, 0.4795],
#        [0.2894, 0.0568, 0.3439],
#        [0.6124, 0.7558, 0.4308]])

#y的值
#tensor([0, 1, 1])




class General_Loss(nn.Module):

    def __init__(self,num_class, **kwargs):
        self.cross=nn.CrossEntropyLoss()
        self.num_class=num_class
    super




















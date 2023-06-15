



import torchvision.models as m
import torch.nn as nn
import torch



class resnet(nn.Module):
    def __init__(self,pretrained=False):
        super(resnet, self).__init__()
        self.model=m.resnet50(pretrained=pretrained)

    def forward(self,x):
        y=self.model(x)
        return y




if __name__ == '__main__':
    x=torch.ones((1,3,224,224))
    x=x.cuda()
    model=resnet(pretrained=False)

    model=model.cuda()

    y=model(x)
    print(y.shape)

















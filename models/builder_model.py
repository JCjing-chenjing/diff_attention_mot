
from models.ResNet.resnet import resnet




def build_model(model_name, **kwargs):


    if model_name == 'resnet':
        pretrained=kwargs.get('pretrained') if kwargs.get('pretrained') else False
        model = resnet(pretrained=pretrained)

    else:
        raise AttributeError ('creating model failed')



    return model









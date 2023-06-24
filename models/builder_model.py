
from models.ResNet.resnet import resnet
from models.diff_track.diff_attention_track import DIFFTrack



def build_model(model_name, args):

    if model_name == 'resnet':
        pretrained=args.pretrained  if args.pretrained else False
        model = resnet(pretrained=pretrained)
    elif model_name == 'diff_track':
        model = DIFFTrack(args)
    else:
        raise AttributeError ('creating model failed')
    return model










import torch



def build_optimizer(optim_name,model,lr,**kwargs):

    optimizer=None
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=5e-4)

    return optimizer










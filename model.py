from torchvision import models
from collections import OrderedDict
from torch import nn
from torch import save as TorchSave
from torch import load as TorchLoad

models = {
        "resnet18" : models.resnet18,
        "alexnet" : models.alexnet,
        "vgg13" : models.vgg13,
        "squeezenet" : models.squeezenet1_0,
        "densenet" : models.densenet161,
        "inception" : models.inception_v3,
        "densenet121" : models.densenet121
}

def model_factory(arch = "densenet121", hidden_units = 512, gpu = False, **kwargs):

    if arch in models.keys():
        model = models[arch](pretrained = True)
    else:
        print(f"Model {arch} is unknown")
        return None

    # make sure to freeze the pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # add our our layer
    model.classifier  = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, hidden_units)), #custom hidden units
                            ('drop1', nn.Dropout(p=0.1)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    if gpu:
        model.to('cuda')
    else:
        model.to('cpu')    

    model.settings = {
        "model" : arch,
        "hidden_units" : hidden_units,
        "gpu" : gpu,
        "class_to_idx" : None,
    }   
    return model

def load(path = "checkpoint.pth"):
    checkpoint = TorchLoad(path)
    model = model_factory(**checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def save(model, path = "checkpoint.pth"):
    checkpoint = {
        "state_dict" : model.state_dict(),
        **model.settings
    }
    TorchSave(checkpoint, path)

# model = model_factory()
# save(model)

# model = load()
# print(model.settings)
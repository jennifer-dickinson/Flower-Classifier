from torchvision import models
from collections import OrderedDict
from torch import nn
from torch import save as TorchSave
from torch import load as TorchLoad

from torch import optim


# models = {
#         "resnet18" : models.resnet18,
#         "alexnet" : models.alexnet,
#         "vgg13" : models.vgg13,
#         "squeezenet" : models.squeezenet1_0,
#         "densenet" : models.densenet161,
#         "inception" : models.inception_v3,
#         "densenet121" : models.densenet121
# }

supported_models = {
    "densenet161" : models.densenet161,
    "vgg16_bn" : models.vgg16_bn,
    "resnet18" : models.resnet18,
    "resnet34" : models.resnet34,
    "resnet50" : models.resnet50,
    "densenet201" : models.densenet201,
    "vgg13_bn" : models.vgg13_bn,
    "densenet121" : models.densenet121,
    "vgg19_bn" : models.vgg19_bn
}

def model_factory(arch = "densenet121", hidden_units = 512, gpu = False, learningrate = 0.001, current_epochs = 0, **kwargs):

    if arch in supported_models.keys():
        model = supported_models[arch](pretrained = True)
    else:
        print(f"Model {arch} is unsupported")
        return None

    # make sure to freeze the pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # add our our layer
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('drop1', nn.Dropout(p=0.1)),
                          ('relu', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.1)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    if gpu:
        model.to('cuda')
    else:
        model.to('cpu')    

    model.settings = {
        "arch" : arch,
        "hidden_units" : hidden_units,
        "gpu" : gpu,
        "class_to_idx" : None,
        "current_epoch" : current_epochs,
        "learningrate" : learningrate
    }   
    return model

def load(path = "checkpoint.pth"):
    checkpoint = TorchLoad(path)
    model = model_factory(**checkpoint)
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["model"])
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def save(model, optimizer, path = "checkpoint.pth"):
    checkpoint = {
        **model.settings,
        "model" : model.state_dict(),
        "class_to_idx" : model.class_to_idx,
        "optimizer" : optimizer.state_dict()
    }
    for key in checkpoint.keys():
        if key != "model" and key != "optimizer":
            print(key, ":", checkpoint[key])
        else:
            print(key, ":", type(checkpoint[key]))
    TorchSave(checkpoint, path)

# model = model_factory()
# save(model)

# model = load()
# print(model.settings)
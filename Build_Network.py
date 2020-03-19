from collections import OrderedDict
from torch import nn
import torchvision.models as models

def build_network(model, hidden, output):
    Input = 25088
    arcch = 'vgg16'
    if model == 'densenet121':
        Input = 1024
        model = models.densenet121(pretrained=True)
        arcch = 'densenet121'
    
    elif model == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    else:
        print("Sorry! Not found\n defualt vgg16")
        model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(Input, hidden),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               
                               nn.Linear(hidden, output),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    return model, arcch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import json
import torchvision.models as models
from PIL import Image

from get_predict_args import get_predict_args

def load_checkpoint(filepath = 'checkpoint.pth', Catt = 'cat_to_name.json'):
    
    checkpoint = torch.load(filepath)
    arcch = checkpoint['arch']
    Input = 0
    if arcch == 'vgg16':
        model = models.vgg16(pretrained=True)
        Input = 25088
    else:
        Input = 1024
        model = models.densenet121(pretrained=True)
    
    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(Input, 255),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               
                               nn.Linear(255, 102),
                               nn.LogSoftmax(dim=1))
    
    classifier.load_state_dict(checkpoint['state_dict'])
    
    model.classifier = classifier
    class_to_idx = model.class_to_idx = checkpoint['class_to_index']
    
    with open(Catt, 'r') as f:
        cat_to_name = json.load(f)

    model.idx_to_class = inv_map = {v: k for k, v in class_to_idx.items()}
    
    for i in model.idx_to_class:
        model.idx_to_class[i] = cat_to_name[model.idx_to_class[i]]
    
    return model

def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model = model.eval()
    inputs = Variable(image.unsqueeze(0))
    inputs = inputs.to(device)
    output = model(inputs)
    
    ps = torch.exp(output).data
    ps_top = ps.topk(topk)
    index_to_class = model.idx_to_class
    

    
    probs = ps_top[0].tolist()[0]
    classes = [index_to_class[i] for i in ps_top[1].tolist()[0]]
    
    return probs, classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    perform_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image = perform_transforms(Image.open(image))
    
    return image

def main():
    in_arg = get_predict_args()
    model = load_checkpoint(in_arg.checkpoint, in_arg.category_names)
    probs, classes = predict(in_arg.path, model, in_arg.top_k, in_arg.gpu)
    
    for i in range(in_arg.top_k):
        print(i+1, ": ", classes[i]," with a percentage of: ",int(probs[i]*100),"%")


    
    
if __name__ == "__main__":
    main()
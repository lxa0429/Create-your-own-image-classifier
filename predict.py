# Basic usage: python predict.py /path/to/image checkpoint
# python predict.py ./flowers/test/10/image_07090.jpg ./checkpoint.pth --gpu True
# python predict.py ./flowers/test/64/image_06138.jpg ./checkpoint.pth --gpu True

import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import helper
from collections import OrderedDict
from workspace_utils import active_session
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from cat_to_flower_name import category_flower_dict

from PIL import Image
import numpy as np

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("imagepath", type=str, default="./image_to_be_tested.jpg", action="store", help="Image path to be tested")
parser.add_argument("checkpoint", type=str, default="./checkpoint.pth", action="store", help="Saved checkpoint path directory")
parser.add_argument("--category_names", type=str, default="cat_to_name.json", action="store", help="File name of mapping of flower categories to real names")
parser.add_argument('--top_k', type=int, default=5, action="store", help="Choose number of topk classes")
parser.add_argument("--gpu", type=str, default=False, action="store", help="Set device to use GPU")

# Read arguments from command line
args = parser.parse_args()

'''
python predict.py /path/to/image checkpoint
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
'''

# Set default value for the following parameters
#checkpoint = 'checkpoint.pth'
#imagepath = './image_to_be_tested.jpg'
#arch = 'vgg16'
#category_names = 'cat_to_name.json'

imagepath = args.imagepath
print('imagepath', imagepath)

checkpoint = args.checkpoint
print('checkpoint', checkpoint)

if args.category_names:
    category_names = args.category_names
    print('category_names', category_names)
if args.top_k:
    topk = args.top_k
    
device = "cpu"
if args.gpu:
    #gpu = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model is running on", device, "device")

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    if checkpoint['arch'] == 'vgg16':    
        in_features = checkpoint['input_size']
        hidden_units = checkpoint['hidden_layer1']
        #print('in_features', in_features)
        #print('hidden_units', hidden_units)
    elif checkpoint['arch'] == 'densenet121':
        in_features = 1024
        hidden_units = 512
   
    for param in model.parameters():
        param.requires_grad = False
        
   
    model.class_to_idx = checkpoint['class_to_idx']
    output_size = checkpoint['output_size']
    
    if checkpoint['arch'] == 'vgg16': 
        hidden_layer2 = checkpoint['hidden_layer2']
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_units, hidden_layer2)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(hidden_layer2, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif checkpoint['arch'] == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model
    

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(image).convert('RGB')
    
    img_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    img_transformed = img_transforms(img)
    # return img_transformed    

    img_np = np.array(img_transformed)
    return img_np


# from torch.autograd import Variable

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    model.eval()
    model = model.to(device)

    # Process image
    image = process_image(image_path)
    
    # Transfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    image = image.to(device)
            
    output = model.forward(image)
    
    # Top probs
    probabilities = torch.exp(output).data
    
    top_probs, top_labs = probabilities.topk(topk)
    
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]   
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]

    return top_probs, top_labels


# Load the checkpoint
trained_model = load_checkpoint(checkpoint)
#print('trained mode', trained_model)
# Predict flower classes
top_probs, top_labels = predict(imagepath, trained_model, topk)
print(top_probs)
print(top_labels)

import json

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
class_string_list = []
for top_label in top_labels:
    for key, value in cat_to_name.items():
        if top_label == key:
            print(key, ": ", value)
            class_string_list.append(value)








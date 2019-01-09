# Imports here

import numpy as np
import json
from collections import OrderedDict
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def main(input):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    category_names = 'cat_to_name.json'
        
    # Load and test network
    checkpoint = '/storage/checkpoint.pth'
    model = load_checkpoint(checkpoint, device)
    
    # Class Prediction
    input_path = 'static/img/' + input
    probs, classes = predict(input_path, model, 1, category_names, device, cat_to_name)
    confidence = probs[0] * 100
    confidence = round(confidence, 2)
        
    return probs, classes, confidence
    
# Functions defined below

def load_checkpoint(filepath, device):
    # Loads a checkpoint and rebuilds the model

    checkpoint = torch.load(filepath)
    
    model = models.vgg19(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Hyperparameters for the network
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = checkpoint['output_size']
    
    # Build a feed-forward network
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_size)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_size, hidden_size)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(hidden_size, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image).convert('RGB')
        
    in_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # Discard alpha channel and add the batch dimension
    image = in_transform(image)
    
    return image.numpy()

def predict(image_path, model, topk, category_names, device, cat_to_name):
    ''' Predict the class (or classes) of an img using a trained deep learning model.
    '''
    
    # Predicts the class from an image file
    img = torch.from_numpy(process_image(image_path)).unsqueeze(0).float()
    
    model, img = model.to(device), img.to(device)
    model.eval()
    model.requires_grad = False
    
    output = torch.exp(model.forward(img)).topk(topk)
    probs, classes = output[0].data.cpu().numpy()[0], output[1].data.cpu().numpy()[0]
    
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    
    if category_names == 'cat_to_name.json':
        for i in np.arange(len(classes)):
            classes[i] = cat_to_name[classes[i]]
    
    return probs, classes
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
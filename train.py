# Imports here

import numpy as np
import argparse
import json
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    data_dir = in_arg.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data, valid_data, test_data, trainloader, validloader, testloader, device = load_data(train_dir, valid_dir, test_dir)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Create model and network
    model = models.__dict__[in_arg.arch](pretrained=True)
    model.classifier = build_network(model, in_arg.hidden_units)
    
    # Train and validate network
    epochs = in_arg.epochs
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    do_deep_learning(model, trainloader, validloader, epochs, criterion, optimizer, device)
    
    # Test network
    validation(model, testloader, criterion)
    check_accuracy_on_test(testloader, model, device)
    
    # Save model as checkpoint
    save_checkpoint(in_arg.hidden_units, model, train_data)
    
# Functions defined below

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 7 command line arguments
    parser.add_argument('data_directory', type=str,
                        help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', 
                        help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=9,
                        help='number of epochs')
    parser.add_argument('--gpu', action='store_const', const=42,
                        help='use GPU for training')

    # returns parsed argument collection
    return parser.parse_args()

def load_data(train_dir, valid_dir, test_dir):
    # Build and train network
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Defines the dataloaders using the image datasets and the transforms
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader, device

def build_network(model, hidden_units):
    # Build and train network

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Hyperparameters for the network
    input_size = 25088
    hidden_size = hidden_units
    output_size = 2

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
    
    return classifier

def validation(model, loader, criterion):
    
    valid_loss = 0
    accuracy = 0
    for images, labels in loader:
        if torch.cuda.is_available:
            model = model.cuda()
            images, labels = images.cuda(), labels.cuda()
            
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def do_deep_learning(model, trainloader, validloader, epochs, criterion, optimizer, device):

    print_every = 40
    steps = 0
    running_loss = 0

    model.to(device)

    for e in range(epochs):
        model.train()

        for images, labels, in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
                
def check_accuracy_on_test(testloader, model, device):

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 250 test images: %d %%' % (100 * correct / total))
    
def save_checkpoint(hidden_size, model, train_data):
    # Save the checkpoint

    checkpoint = {'input_size': 25088,
                  'output_size': 2,
                  'hidden_size': hidden_size,
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    # Call to main function to run the program
if __name__ == "__main__":
    main()
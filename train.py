# Imports required packages
import argparse
#import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import helper
from collections import OrderedDict
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from workspace_utils import active_session
import json


def get_datasets(args):
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validate_datasets = datasets.ImageFolder(valid_dir, transform=validate_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    validateloader = torch.utils.data.DataLoader(validate_datasets, batch_size=64)

    return [train_datasets, test_datasets, validate_datasets]


def build_model(args, image_datasets):
    device = 'cpu'
    if(args.hidden_units):
        hidden_units = args.hidden_units
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arch = args.arch
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    if arch.lower() == 'vgg16':
        in_features = model.classifier[0].in_features
        if hidden_units < 4096:
            hidden_units = 4096
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_units, 256)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif arch.lower() == 'densenet121':
        in_features = 1024
        if hidden_units > 512:
            hidden_units = 512
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    #image_datasets = [train_datasets, validate_datasets, test_datasets]
    model.class_to_idx = image_datasets[0].class_to_idx

    print("Model is built.")
    return model, optimizer, criterion, in_features, hidden_units

def train_model(model, optimizer, criterion, device, image_datasets):
    print("Training model now...")
    steps = 0
    running_loss = 0
    print_every = 5
    trainloader = torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets[1], batch_size=64)
    #validateloader = torch.utils.data.DataLoader(image_datasets[2], batch_size=64)

    train_losses, test_losses = [], []

    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
            
                optimizer.zero_grad()
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                        
                            test_loss += batch_loss.item()
                        
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                    train_losses.append(running_loss/len(trainloader))
                    test_losses.append(test_loss/len(testloader))
                        
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(testloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()
                    
    print("Model is trained")

def save_model(model, in_features, hidden_units, datasets, save_dir):
    model.class_to_idx = datasets[0].class_to_idx

    if arch.lower() == 'vgg16':
        checkpoint = {'input_size': in_features,
                    'hidden_layer1':hidden_units,
                    'hidden_layer2': 256,
                    'output_size': 102,
                    'arch': arch,
                    'learning_rate': lr,
                    'batch_size': 64,
                    'epochs': epochs,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx}
    
    elif arch.lower() == 'densenet121':
        in_features = 1024
        if hidden_units > 512:
            hidden_units = 512
        checkpoint = {'input_size': in_features,
                    'hidden_layer1':hidden_units,
                    'output_size': 102,
                    'arch': arch,
                    'learning_rate': lr,
                    'batch_size': 64,
                    'epochs': epochs,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx}
    
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print('Model is saved in ', save_dir + '/checkpoint.pth')

def parsing_cmdline_args():
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("data_directory", default="flowers", action="store", help="Set data directory for images")
    parser.add_argument("--arch", type=str, default='densenet121', action="store", help="Select architecture model")
    parser.add_argument("--learning_rate", type=float, default=0.001, action="store", help="Set learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, action="store", help="Set hidden units")
    parser.add_argument("--epochs", type=int, default=5, action="store", help="Set number of epoch")
    parser.add_argument("--save_dir", type=str, default=".", action="store", help="Set directory to save check point")
    parser.add_argument("--gpu", type=str, default=False, action="store", help="Set device to use GPU")

    # Read arguments from command line
    args = parser.parse_args()
    return args

# vgg16, densenet121
# python train.py flowers --arch vgg16 --gpu True --epochs 2
# python train.py flowers --arch densenet121 --gpu True
if __name__ == '__main__':

    args = parsing_cmdline_args()

    if args.arch:
        print("--arch as: % s" % args.arch)
        arch = args.arch
    if args.learning_rate:
        print("--learning_rate as: % s" % args.learning_rate)
        lr = args.learning_rate
    if args.hidden_units:
        print("--hidden_units as: % s" % args.hidden_units)
        hidden_units = args.hidden_units
    if args.epochs:
        print("--epochs as: % s" % args.epochs)
        epochs = args.epochs
    if args.save_dir:
        save_dir = args.save_dir
        
    data_dir = args.data_directory
    device = 'cpu'
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model is running on", device, "device")

#-------------------------------------------------------------------------------------
    # Prepare data
    datasets = get_datasets(args)
    # Build model
    model, optimizer, criterion, in_features, hidden_units = build_model(args, datasets)
    # Train model
    train_model(model, optimizer, criterion, device, datasets)
    # Save the train model
    save_model(model, in_features, hidden_units, datasets, save_dir )
#---------------------------------------------------------------------------------------
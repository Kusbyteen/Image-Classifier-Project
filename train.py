import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import seaborn as sns 
import json
from collections import OrderedDict
from arg_parse import get_train_args
import os

train_args = get_train_args()

data_dir = train_args.data_dir
train_dir = train_args.data_dir + '/train'
valid_dir = train_args.data_dir + '/valid'
test_dir = train_args.data_dir + '/test'
# print(os.listdir(data_dir))


if train_args.gpu == 'yes':
    train_args.device = 'cuda'
    print('you are using gpu')
elif train_args.gpu == 'no':
    train_args.device = 'cpu'
    print('you are using cpu')
    
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = torchvision.transforms.Compose([
                                       torchvision.transforms.RandomRotation(45),
                                       torchvision.transforms.RandomResizedCrop(224), 
                                       torchvision.transforms.RandomHorizontalFlip(),
                                       torchvision.transforms.ToTensor(),
                                       normalize,])
valid_transforms = torchvision.transforms.Compose([ torchvision.transforms.Resize(256),
                                        torchvision.transforms.RandomCrop(224), 
                                        torchvision.transforms.ToTensor(),
                                        normalize,])
train_datasets = torchvision.datasets.ImageFolder(train_dir, transform=data_transforms)
valid_datasets = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = torchvision.datasets.ImageFolder(test_dir, transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

with open(train_args.cat_to_name, 'r') as f:
    train_args.cat_to_name = json.load(f)
if train_args.arch   == 'vgg19' or train_args.arch == 'vgg':
    model = torchvision.models.vgg19(pretrained=True)
    train_args.input_size = 25088 
    
elif train_args.arch == 'densenet121' or train_args.arch == 'densenet':
    model = torchvision.models.densenet121(pretrained=True)
    train_args.input_size = 1024
else:
    print("Please the application will use the default arch")

for param in model.parameters():
    param.requires_grad = False

classifier = torch.nn.Sequential(OrderedDict([
                                      ('fc1', torch.nn.Linear(train_args.input_size, train_args.hidden_units,  bias=True)),
                                      ('Relu1', torch.nn.ReLU()),
                                      ('Dropout1', torch.nn.Dropout(p = 0.5)),
                                      ('fc2', torch.nn.Linear(train_args.hidden_units, 102,  bias=True)),
                                      ('output', torch.nn.LogSoftmax(dim=1))
                                       ]))

model.classifier = classifier
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=train_args.lr)

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(train_args.device), labels.to(train_args.device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

epochs = train_args.epochs
running_loss = 0

for e in range(epochs):
    model.train()
    model.to(train_args.device)
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(train_args.device), labels.to(train_args.device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(model, valid_loader, criterion)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/100),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

            running_loss = 0
            model.train()
            
with torch.no_grad():   
    test_loss, accuracy = validation(model, test_loader, criterion)
    print("Validation Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
          "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)))
    
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epoch': epochs,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer': optimizer.state_dict()
              }

torch.save(checkpoint, train_args.saved_model_path)


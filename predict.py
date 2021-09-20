import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import seaborn as sns 
import json
from collections import OrderedDict
import PIL as p
import os
from arg_parse import get_predict_args


predict_args = get_predict_args()

if predict_args.gpu == 'yes':
    predict_args.device = 'cuda'
elif predict_args.gpu == 'no':
    predict_args.device = 'cpu'
    
def load_checkpoint(filename=predict_args.saved_model_path):
    
    checkpoint = torch.load(filename)
    if predict_args.arch   == 'vgg19' or predict_args.arch == 'vgg':
        model = torchvision.models.vgg19(pretrained=True)
        predict_args.input_size = 25088 

    elif predict_args.arch == 'densenet121' or predict_args.arch == 'densenet':
        model = torchvision.models.densenet121(pretrained=True)
        predict_args.input_size = 1024
    else:
        print("Please the application will use the default arch")

    
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(predict_args.input_size, predict_args.hidden_units,  bias=True)),
                                                ('Relu1', torch.nn.ReLU()),
                                                ('Dropout1', torch.nn.Dropout(p = 0.5)),
                                                ('fc2', torch.nn.Linear(predict_args.hidden_units, 102,  bias=True)),
                                                ('output', torch.nn.LogSoftmax(dim=1))
                                                 ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = p.Image.open(image_path)
    width , height = image.size
    image.thumbnail((1000000, 256)) if width > height else image.thumbnail((256, 200000)) 
    width, height = image.size
    left = (width - 224)/2.
    top = (height - 224)/2.
    right = (width + 224)/2.
    bottom = (height + 224)/2.
    im = image.crop((left, top, right, bottom))
    np_im = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - mean)/std
    np_im = np_im.transpose((2, 0, 1))
    return np_im

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
image_path = predict_args.image_path

def predict(image_path, model, topk=predict_args.topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    img = torch.from_numpy(image).type(torch.FloatTensor)
    image = img.unsqueeze(0)
    image = image.float().cuda()
    
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        model = model.double()
        output = model.forward(image.double())
    
    ps = torch.exp(output)
    
    probs, classes = torch.topk(ps, topk)
    
    probs = probs.cpu().numpy().tolist()[0]
    classes = classes.cpu().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    
    with open(predict_args.cat_to_name, 'r') as f:
            predict_args.cat_to_name = json.load(f)
            
    classes = [idx_to_class[x] for x in classes]
    flowers = [predict_args.cat_to_name[str(x)] for x in classes]
    
    return probs, flowers


probs, flowers = predict(image_path, model, topk=predict_args.topk)

def view_classify(image_path, probs, flowers):  
    ''' Function for viewing an image and it's predicted classes.
    '''
    df = pd.DataFrame({'ps':probs, 'clas': flowers})
    im= p.Image.open(image_path)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(np.array(im))
    plt.show()
    plt.axis('off')
    plt.subplot(2,1,2)
    sns.barplot(x="ps", y="clas", data=df, orient = "h")   
    plt.tight_layout()
    plt.show()

print(flowers, probs)  
view_classify(image_path, probs, flowers)
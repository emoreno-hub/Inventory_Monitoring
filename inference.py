import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ]

def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features

    model.fc = nn.Sequential( nn.Linear(num_features, 512),
                             nn.BatchNorm1d(512),
                             nn.Dropout(0.2),
                             nn.Linear(512, 256),
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 5), # set to 5 classes
                            )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model




def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # Process an image uploaded to the endpoint
    logger.info(f'Incoming Requests Content-Type is: {content_type}')
    logger.info(f'Request body Type is: {type(request_body)}')
    if content_type in ACCEPTED_CONTENT_TYPE:
        logger.info(f"Returning an image of type {content_type}" )
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested an unsupported Content-Type: {content_type}, Accepted Content-Type are: {ACCEPTED_CONTENT_TYPE}")

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import smdebug.pytorch as smd
import logging
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    logger.info(f'Start Testing')
    model.eval()

    running_loss=0
    running_corrects=0
    
    with torch.no_grad(): # turn off gradients during testing
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        
        logger.info(f"Testing Loss: {total_loss}")
        logger.info(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, validation_loader, criterion, epochs, optimizer, device):
    # set model to train
    model.train()
    # start logging
    logger.info("Start Training")
    
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))

        if loss_counter==1:
            break
            
    return model
    
def net():
    model = models.resnet50(pretrained=True) # instantiate pretrained resnet50 model
    
    for param in model.parameters():
        param.requires_grad = False # freeze the convolutional layers of the pretrained layers by setting to false
    
    # find the number of features present in the pretrained model
    num_features = model.fc.in_features
    
    # add a fully connected layer to the end of our model
    model.fc = nn.Sequential( nn.Linear(num_features, 512),
                             nn.BatchNorm1d(512),
                             nn.Dropout(0.2),
                             nn.Linear(512, 256),
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 5), # set to 5 classes
                            )
    return model

def create_data_loaders(train_path, test_path, valid_path, batch_size):
    logger.info("Get train and test data loader")
    
    training_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.RandomRotation(10), # rotate by 10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # no augmentation on test / validation set; only resize and crop
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    validset = torchvision.datasets.ImageFolder(root=valid_path, transform=testing_transform)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size) 

    return train_loader, test_loader, valid_loader

def main(args):
    # move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # log device, hyperparameters, model path, and output path
    logger.info(f"Running on Device {device}")
    logger.info(f"Hyperparameters : LR: {args.lr}, Batch Size: {args.batch_size}, Epoch: {args.epochs}")
    logger.info(f"Model Dir Path: {args.model_dir}")
    logger.info(f"Output Dir Path: {args.output_dir}")

    # initialize model
    model=net()
    model = model.to(device)

    #  create the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    # get training and test loaders
    train_loader, test_loader, validation_loader = create_data_loaders(args.train_data, args.test_data, args.valid_data, args.batch_size )

        
    logger.info(f"Begin Training")
    model=train(model, train_loader, validation_loader, loss_function, args.epochs, optimizer, device)

    logger.info(f"Begin Testing")
    test(model, test_loader, loss_function, device)
    
    # Save the trained model and capture in logging
    logger.info("Saving The Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    logger.info("Model Saved")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    # create arguments for CNN model
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=4, metavar="N", help="number of epochs to train (default: 4)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 0.1)" )
                        
    # Use SageMaker OS environment channels to locate data, model directory, and output directory to save in S3 bucket
    parser.add_argument('--train_data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test_data', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--valid_data', type=str, default=os.environ['SM_CHANNEL_VALIDATE'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)

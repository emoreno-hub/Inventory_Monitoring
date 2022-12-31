# Inventory Monitoring at Distribution Centers
In this project we use a Convolutional Neural Network that relies on a pre-trained ResNet50 model to count the number of objects in each bin containing varying quantities of products.  These bins are commonly used within distribution centers to carry product throughout the operation.  The goal with ensuring accurate counting of items in bins is to improve inventory management.  Regardless of a company’s supply chain being large or small, inventory management is vital to a company’s health because it balances supply with demand by ensuring that product is available at the right time by tracking product up and down the supply chain.  Too much stock costs money and reduces cash flow and too little stock could lead to unfilled customer orders and lost sales.


## Project Set Up and Installation
- Create a SageMaker domain
- Open SageMaker Studio and select an instance type (ml.t3.medium will do)
- Clonse this repository
- Run `sagemaker.ipnyb`

## Dataset

### Overview
The dataset used for this project will be the Amazon Bin Image Dataset, which consists of 500,000 images of bins containing one or more objects from Amazon Fulfillment Centers.  Each image is in JPEG format and contains corresponding JSON metadata files which describe the items in each bins.  

### Access
Running the `sagemaker.ipynb` notebook will allow you to download a subset of data.  The subset is controlled by the file titled `file_list.json` and reuslts in 10,441 images being downloaded for this project.

## Model Training

#### CNN Model
This project will used a Convolutional Neural Network (CNN) which is a deep learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects / objects in the image, and be able to differentiate one from the other.  A pre-trained ResNet50 model was used to build the CNN model with a fully connected layer added to the end of the model followed by a ReLu activation function, then another Linear layer, and a final ReLu activation function.

Batch normalization is a technique for training very deep neural networks that normalizes the contributions to a layer for every mini-batch. This has the impact of settling the learning process and drastically decreasing the number of training epochs required to train deep neural networks, which is why it was included.

```
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
                             nn.Linear(256, 5), # output of model is 5 classes
                            )
    return model
```


#### Data Augmentation
Minor data augmentation was performed on the training images by performing a random horizontal flip along with resizing and cropping.  No augmentation was performed on the validation and test images, only resizing and cropping.

```
    training_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.RandomRotation(10), # rotate by 10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
```

#### Train-Validation-Test Split
The data was split into train, validation, and test sets using an 80/10/10 split, respectively.  

![](https://github.com/emoreno-hub/Inventory_Monitoring_Project/blob/main/screenshots/Dataset_image_distribution.png)

#### Hyperparameter Tuning
Hyperparameter tuning was performed to find the optimal parameters across the below search space.  There seems to be a consensus among many that low batch sizes perform better than large batch sizes, which is why I opted for batch sizes of 32 and 64.  Training was done for epochs ranging from 10 to 20.

In order to speed up training, `max_jobs` was set to 4 and `max_parallel_jobs` was set to 2.

```
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64]),
    "epochs": IntegerParameter(10,20)
}
```

![](https://github.com/emoreno-hub/Inventory_Monitoring_Project/blob/main/screenshots/Training_jobs.PNG)


After hyperparameter tuning concluded, the optimal hyperparameters found were:

```
{'batch_size': 64, 'lr': 0.0014671136028592893, 'epochs': 20}
```

A moodel was then trained using these hyperparameters using `ml.g4dn.xlarge` for the instance type.

#### Model Evaluation
By instantiating a pre-trained ResNet50 model with the optimal parameters found from hyper parameter tuning, accuracy for this model was 30%, which is lower than the benchmark of 55.67%.  It could be the case that I added too many fully connected layers at the end of the model, which I will explore down the road.

![](https://github.com/emoreno-hub/Inventory_Monitoring_Project/blob/main/screenshots/Model_performance.PNG)

## Machine Learning Pipeline
The AWS machine learning pipeline used for this project is as follows:
- Data collection
- Data preprocessing
- Hyperparameter tuning
- Model evaluation
- Model deployment

### Model Deployment

After the model was trained, a PyTorch estimator was deployed to an endpoint using `ml.m5.large` for the instance type and predictions were made using sample test images.

![](https://github.com/emoreno-hub/Inventory_Monitoring_Project/blob/main/screenshots/Prediction.PNG)

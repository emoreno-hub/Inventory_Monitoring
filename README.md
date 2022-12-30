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


#### Data Augmentation
Minor data augmentation was performed on the training images by performing a random horizontal flip along with resizing and cropping.  No augmentation was performed on the validation and test images, only resizing andf cropping.

#### Train-Validation-Test Split
The data was split into train, validation, and test sets using an 80/10/10 split, respectively.  

![](https://github.com/emoreno-hub/Inventory_Monitoring_Project/blob/main/screenshots/Dataset_image_distribution.png)

#### Hyperparameter Tuning
Instead of relying on specified hyperparameters, hyperparameter tuning was performed to across the below search space.  There seems to be a consensus among many that low batch sizes perform better than large batch sizes, which is why I opted for batch sizes of 32 and 64.

In order to speed up training, `max_jobs` was set to 4 and `max_parallel_jobs` was set to 2.

```
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64]),
    "epochs": CategoricalParameter([8,12,15])
}
```

![](https://github.com/emoreno-hub/Inventory_Monitoring_Project/blob/main/screenshots/Training_jobs.PNG)


After hyperparameter tuning concluded, the optimal hyperparameters found were:

```
{'batch_size': 32, 'lr': 0.003309487406915597, 'epochs': 12}
```

A moodel was then trained using these hyperparameters using `ml.g4dn.xlarge` for the instance type.

#### Model Evaluation
Unfortunately, despite using a pre-trained ResNet50 model and hyperparameter tuning, accuracy for this model was only 11.8%.  This is significantly lower than the benchmark of 55.67%.  It could be the case that I added too many fully connected layers at the end of the model, which I will explore down the road.

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

# Imports here
import torch
from torch import nn as nn
from torch import optim as optim
import nnModel
import helpers.DataLoader
import argparse

#######################################################
# Train a Neural Network using transfer learning:
# 1. Get the directory to the image files to train with
# 2. Set the directory to save checkpoints
# 3. Choose the architecture
# 4. Set the hyperparameters
# 5. Choose GPU for training

# Create the parser and add the arguments
parser = argparse.ArgumentParser(description="Train a Neural Network using transfer learning")
# 1. Get the directory to the image files to train with
parser.add_argument('data_directory', 
                    help="The relative path to the image files to train on. It should include two folders: 'train' and 'test' for training.")
# 2. Get the directory to the image files to train with
parser.add_argument('--save_dir', default='/',
                    help="The relative path to save the neural network checkpoint")
# TODO: Add more support for other architectures                    
# 3. Choose the architecture
parser.add_argument('--arch', default="vgg19",
                    help="The architecture you wish to train the model with. Can be: vgg19 or densenet161")
# 4. Set the hyperparameters: Learning Rate, Hidden Units, Training Epochs, Training batch size
parser.add_argument('--learning_rate', type=float, default="0.01",
                    help="The learning rate for the model")
parser.add_argument('--hidden_units', type=int, default=512,
                    help="The number of units in the hidden layer")
parser.add_argument('--epochs', type=int, default=15,
                    help="The amount of training epochs you wish to use")
parser.add_argument('--batch_size', type=int, default=32,
                    help="The size of the batches you want to use for training")
# 5. Choose the GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="If you would like to use the GPU for training. Default is False")

# Collect the arguments
args = parser.parse_args()
data_directory = args.data_directory
save_directory = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
batch_size = args.batch_size
gpu = args.gpu

# Get the image data from the files and create the data loaders
train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets = helpers.DataLoader.load_image_data(data_directory, batch_size)
    
# Create the model. Returns 0 if model cant be created
model = nnModel.create_model(arch, hidden_units)

# If we sucessfully create a model continue with the training
if model != 0:
    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    # TODO Maybe move this to a select optimizer function
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate) 

    # Train the model with validation
    nnModel.train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs, gpu)

    # Save the model
    nnModel.save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch)

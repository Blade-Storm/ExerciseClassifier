import nnModel
import torchvision
from torchvision import datasets, transforms, models
import helpers.JsonLoader
import argparse


#######################################################
# Load a saved model and predict a flower image
# 1. Get the directory to the image that we will infer on
# 2. Get the directory to the save checkpoint
# 3. Get the number of most likley classes to show
# 4. Get the name of the json file to map categories to real names
# 5. Choose GPU for inference

# Create the parser and add the arguments
parser = argparse.ArgumentParser(description="Load a Neural Network to use for inference")
parser.add_argument('data_directory', 
                    help="The relative path to the image files to infer on.")
parser.add_argument('checkpoint', 
                    help="The relative path to the neural network checkpoint as a pth file.")
parser.add_argument('--top_k', default=1, type=int,
                    help="The amount of most likley classes to return for the predictions")
parser.add_argument('--category_names', default = './cat_to_name.json',
                    help="The json file (including file path) to load category names")
parser.add_argument('--gpu', default=False, action='store_true',
                    help="Boolean to infer with the gpu")

# Collect the arguments
args = parser.parse_args()
data_directory = args.data_directory
checkpoint = args.checkpoint
top_k = args.top_k
category_name = args.category_names
use_gpu = args.gpu

# Load the model
model = nnModel.load_model(checkpoint)

# Load the content of the json file
categories = helpers.JsonLoader.load_json(category_name)

# Predict
nnModel.predict(categories, data_directory, model, use_gpu, top_k)

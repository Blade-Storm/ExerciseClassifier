import cv2
import numpy
import os
import torchvision
from torchvision import transforms
from PIL import Image



# Directory where p or n files are at
data_dir = './p/'
#data_dir = './n'

name = "shopping.jpg"

# Open the image
#mage = Image.open(data_dir + name)

# Transform the image size
#transform = transforms.Compose([transforms.Resize(65),
#                                    transforms.CenterCrop(50),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize([0.485, 0.456, 0.406],
#                                                        [0.229, 0.224, 0.225])])
#image = transform(image).float()
image = cv2.imread(data_dir + name)

image = cv2.resize(image, (32,32))
# Save the new image
# Save the image of the current frame
name = "image" + str(3) +  '.jpg'
print('Saving...' + name)
cv2.imwrite(name, image)
import os
import cv2
import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import helpers.ProcessImage
import matplotlib.pyplot as plt
import numpy as np
import win32api
from win32api import GetSystemMetrics

#from workspace_utils import active_session





def create_model(arch, hidden_units):
    # Load a pretrained network (densenet161)
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    print("Creating the model...")
    # Load a pretrained model
    if arch.lower() == "vgg19":
        model = models.vgg19(pretrained=True)
        input_features = 25088
    elif arch.lower() == "densenet161":
        model = models.densenet161(pretrained=True)
        input_features = 2208
    else:
        # We dont support the entered model architecture so return to start over
        print("Model architecture: {} is not supported. \n Try vgg19 or densenet161".format(arch.lower()))
        return 0
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model
    model.classifier = nn.Sequential(nn.Linear(input_features,hidden_units),
                                     nn.ReLU(),         
                                     nn.Dropout(0.5),                            
                                     nn.Linear(hidden_units,3),
                                     nn.LogSoftmax(dim=1))

    print("Done creating the model\n")
    return model



def train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs, use_gpu):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    # Track the loss and accuracy on the validation set to determine the best hyperparameters
    print("Training the model...\n")

    # Use the GPU if its available
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set the model to the device for training
    model.to(device)

    # Capture the current time for tracking purposes
    start_time = time.time()

    # With an active session train our model
    #with active_session():
    train_losses, validation_losses, training_accuracies, validation_accuracies = [], [], [], []
    

    # Create the training loop
    for e in range(epochs):
        # Set the model back to train mode
        model.train()
        # Define the training loss for each epoch
        training_loss = 0

        training_accuracy = 0
        
        for images, labels in train_dataloaders:            
            # Move the image and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients since they accumulate
            optimizer.zero_grad()
            
            # Get the log probability from the model
            logps = model.forward(images)

            # Get the loss
            loss = criterion(logps, labels)

            # Backpropagate
            loss.backward()

            # Gradient Descent
            optimizer.step()

            # Keep track of the training loss
            training_loss += loss.item()

            # Get the top class from the predictions
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            # Get the accuracy for the prediction
            training_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        else:
            # Keep track of the validation loss and accuracy
            validation_loss = 0
            validation_accuracy = 0
            
            #print("validation")
            
            # Set the model to evaluation mode. This will turn off the dropout functionality
            model.eval()

            # Turn off the gradients for validation
            with torch.no_grad():
                # Create the validation loop
                for images, labels in valid_dataloaders:
                    # Move the image and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)
                    
                    #images.resize_(images.shape[0], -1)
                    
                    # Get the log probability 
                    logps = model.forward(images)
                    
                    # Get the loss
                    loss = criterion(logps, labels)
                    
                    # Get probability from the model
                    ps = torch.exp(logps)
                    
                    # Get the top class from the predictions
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    # equals = (labels.data == ps.max(1)[1])
                    # Get the accuracy for the prediction
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # Keep track of the validation loss
                    validation_loss += loss.item()
                    
        # Get the total time that has elapsed
        elapsed_time = time.time() - start_time  

        # Update the training and validation losses to graph the learning curve
        train_losses.append(training_loss/len(train_dataloaders))
        validation_losses.append(validation_loss/len(valid_dataloaders))
        training_accuracies.append(training_accuracy/len(train_dataloaders) * 100)
        validation_accuracies.append(validation_accuracy/len(valid_dataloaders) * 100)

        # Print out the statistical information
        print("Training Epoch: {}\n".format(e),
                "Training Loss: {}\n".format(training_loss/len(train_dataloaders)),
                "Training Accuracy: {}\n".format(training_accuracy/len(train_dataloaders) * 100),
                "Validation Loss: {}\n".format(validation_loss/len(valid_dataloaders)),
                "Validation Accuracy: {}\n".format(validation_accuracy/len(valid_dataloaders) * 100),
                "Total Time: {}\n".format(elapsed_time))  

    
    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.legend(frameon=False)
    plt.show()
    print("\nDone training the model \n")



def save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch):

    print("Saving the model...")
    # Before saving the model set it to cpu to aviod loading issues later
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)

    # Save the train image dataset
    model.class_to_idx = train_datasets.class_to_idx

    if arch.lower() == "vgg19":
        input_features = 25088
    elif arch.lower() == "densenet161":
        input_features = 2208

    # Save other hyperparamters
    # TODO: Pass in the input size based on the model
    checkpoint = {'input_size': input_features,
                'output_size': 102,
                'hidden_units': hidden_units,
                'arch': arch,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'classifier' : model.classifier,
                'epochs': epochs,
                'criterion': criterion,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}


    torch.save(checkpoint, 'checkpoint.pth')
    print("Done saving the model")


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_file):
    print("Loading the model...")
    # Load the model and force the tensors to be on the CPU
    checkpoint = torch.load(checkpoint_file,  map_location=lambda storage, loc: storage)
   
    if(checkpoint['arch'].lower() == 'vgg19'):
        model = models.vgg19(pretrained=True)
    elif(checkpoint['arch'].lower() == 'densenet161'):
        model = models.densenet161(pretrained=True)

    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    print("Done loading the model")
    return model    




def predict(categories, image_path, model, use_gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use the GPU if its available
    #device = torch.device('cuda' if use_gpu else 'cpu')
    model.to('cpu')
    
    #Switch the model to evaluation mode to turn off dropout
    model.eval()
    
    with torch.no_grad():
        # Get the file extension
        _, file_extension = os.path.splitext(image_path)

        # Determine if the file is an image or video
        # If the file is a video use VideoCapture logic to get each frame in the video and track the weight to count reps
        # Return the reps, class and probabilities
        # TODO: Figure out a more generic way to get the color of the wieghts. Perhaps train a classifier to return the RGB values to then conver to HSV
        if file_extension == '.mp4':
            # Get the vieo from a mp4 file
            video_capture = cv2.VideoCapture(image_path)

            # Get how many frames are in the video to classify the exersice once at the start, middle, and end of the video
            # Set the frame_count to 
            frame_count = 1 
            # lets try to determine the number of frames in a video via video properties; this method can be very buggy and might throw an error based on the OpenCV version
            # or may fail entirely based on which video codecs you have installed
            try:
                frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))    
            # Error
            except:
                print("There was an issue counting the frames in the video")

            #def nothing(x):
            #    pass

            #cv2.namedWindow("Tracking Window")
            #cv2.createTrackbar("Lower Hue", "Tracking Window", 0, 255, nothing)
            #cv2.createTrackbar("Lower Saturation", "Tracking Window", 0, 255, nothing)
            #cv2.createTrackbar("Lower Value", "Tracking Window", 0, 255, nothing)
            #cv2.createTrackbar("Upper Hue", "Tracking Window", 255, 255, nothing)
            #cv2.createTrackbar("Upper Saturation", "Tracking Window", 255, 255, nothing)
            #cv2.createTrackbar("Upper Value", "Tracking Window", 255, 255, nothing)

    
            first_point = 0
            reps = 0
            left_range = False
            completed_rep = False
            # Loop through the video and track the weights to count the reps
            while True:   
                # Capture the next frame
                _, frame = video_capture.read()
                #frame = cv2.imread("./ImagesForProject/test/3/image3771.jpg")

                # If there are no more frames to capture break out of the loop
                if frame is None:
                    break

                
                # Convert the colored image from the video into an HSV image
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #l_h = cv2.getTrackbarPos("Lower Hue", "Tracking Window")
                #l_s = cv2.getTrackbarPos("Lower Saturation", "Tracking Window")
                #l_v = cv2.getTrackbarPos("Lower Value", "Tracking Window")

                #u_h = cv2.getTrackbarPos("Upper Hue", "Tracking Window")
                #u_s = cv2.getTrackbarPos("Upper Saturation", "Tracking Window")
                #u_v = cv2.getTrackbarPos("Upper Value", "Tracking Window")

                # Define the color of the weights to track
                # TODO: Figure out a more generic way to get the color of the wieghts. Perhaps train a classifier to return the RGB values to then convert to HSV
                lower_color_bound = np.array([173 , 56, 16])
                upper_color_bound = np.array([248,213,73])

                # Create the mask with the upper and lower color bounds
                mask = cv2.inRange(hsv, lower_color_bound, upper_color_bound)

                # Get the points the correspond with the colors we are looking for
                points = cv2.findNonZero(mask)
                
                # Get the avg of the points locations, this will allow us to center onto the color we are looking for
                avg = np.mean(points, axis=0)
                
                # Get the height and the width of the frame to get the point where the object is on the screen
                # Get the points corrdinates on the frame
                frame_height, frame_width = frame.shape[:2]
                # Get the width and height of the current screen
                width  = int(GetSystemMetrics(0))
                height = int(GetSystemMetrics(1))

                # Get the point where the weights are
                point_in_screen = ((width/frame_width * avg[0][0]), (height/frame_height * avg[0][1]))
                
                # Store the starting point
                if first_point == 0:
                    first_point = point_in_screen[1]
                
                # Classify the exercise and determine the reps
                # Classify once at the begning, middle, and end of the video
                # CAP_PROP_POS_FRAMES = 1
                if video_capture.get(1) == 1 or video_capture.get(1) == frame_count / 2 or video_capture.get(1) == frame_count:
                    # Processs the image
                    image_tensor = helpers.ProcessImage.process_image(frame, False)

                    # unsqueeze the tensor and set it to the device for inference            
                    image_tensor = image_tensor.unsqueeze_(0)
                    image = Variable(image_tensor)
                    image.to('cpu')
 
                    # Use the model to make a prediction
                    logps = model.forward(image)
                    ps = torch.exp(logps)
                    
                    # Get the top 5 probabilities and index of classes. This is returned as a tensor of lists
                    p, classes = ps.topk(1)
                    
                    # Get the first items in the tensor list to get the list of probs and classes
                    top_p = p.tolist()[0]
                    top_classes = classes.tolist()[0]
                    #print("Top p: {}".format(top_p))
                    #print("Classes: {}".format(top_classes))
                    
                    
                    # Get the exersice name from the json file, using the indexes that come back from topk, and store them in a list
                    labels = []
                    for c in top_classes:
                        labels.append(categories[str(c+1)])

                    # Zip the probabilities with the exersice name
                    output = list(zip(top_p, labels))

                    print("Top Class and Probability: {} {}".format(output[0][1], output[0][0] * 100))
                    

                # For Exersices that start with the weight at the top and move: TOP -> BOTTOM -> TOP to count one repitition:
                # If the current point is less than the starting point we are still at the top of the movement
                # Weight starts at the top where numbers are close to 0 and then moves down increasing disctance from x-axis
                # The + 40 is abritrary and gives us a "range" for when there is variance when a person is moving but not starting the exercise movement
                # Leaving and re-entering this range will determine a repitition
                if point_in_screen[1] < first_point + 40:
                    # If left_range is true then we have returned from the bottom of the movement 
                    # Set left_range back to False since we are back at the top of the movement
                    # Incrament rep
                    if left_range == True:
                        completed_rep = True
                        left_range = False
                        reps += 1
                        print("Total Reps: {}".format(reps))
                # Weight is currently out of the top of the movement and has started to go or is coming back from the bottom
                elif point_in_screen[1] > first_point + 40:
                    # If left_range is False then we have just started the movement to go down
                    # Set completed_rep to False
                    # Set left_range to True 
                    if left_range == False:
                        completed_rep = False
                        left_range = True
                        

                #print("point_in_screen: {}".format(point_in_screen[1]))
                #print("first_point: {}".format(first_point))

                # For viewing purposes when using an image or video
                #res = cv2.bitwise_and(frame, frame, mask = mask)
                #cv2.imshow("frame", frame)
                #cv2.imshow("mask", mask)
                #cv2.imshow("res", res)
                
                #key = cv2.waitKey(1)
                #if key == 27:
                #    break

            # Release the capture and kill any cv windows
            video_capture.release()
            cv2.destroyAllWindows()


        
        elif file_extension == '.jpg':
        # If the file is an image predict the class and return the class and probabilities
            # Processs the image
            image_tensor = helpers.ProcessImage.process_image(image_path, True)

            # unsqueeze the tensor and set it to the device for inference            
            image_tensor = image_tensor.unsqueeze_(0)
            image = Variable(image_tensor)
            image.to('cpu')

            # Use the model to make a prediction
            logps = model.forward(image)
            ps = torch.exp(logps)
            
            # Get the top 5 probabilities and index of classes. This is returned as a tensor of lists
            p, classes = ps.topk(topk)
            
            # Get the first items in the tensor list to get the list of probs and classes
            top_p = p.tolist()[0]
            top_classes = classes.tolist()[0]
            print("Top p: {}".format(top_p))
            print("Classes: {}".format(top_classes))
            
            
            # Get the exersice name from the json file, using the indexes that come back from topk, and store them in a list
            labels = []
            for c in top_classes:
                labels.append(categories[str(c+1)])

            # Zip the probabilities with the exersice name
            output = list(zip(top_p, labels))

            print("Top Probabilities and their Classes: {}".format(output))
            return top_p, labels
        else:
        # Return if the file is not a video or image format we support
            print("The image or video file was not in a supported format. Try jpg or mp4")
            return 0
        
        

        

def sanity_check(cat_to_name, file_path, model, index):
    # Display an image along with the top 5 classes
    # Create a plot that will have the image and the bar graph
    fig = plt.figure(figsize = [10,5])

    # Create the axes for the flower image 
    ax = fig.add_axes([.5, .4, .225, .225])

    # Process the image and show it
    result = helpers.ProcessImage.process_image(file_path, True)
    ax = helpers.ProcessImage.imshow(result, ax)
    ax.axis('off')

    ax.set_title(cat_to_name[str(index)])


    # Make a prediction on the image
    predictions, classes = predict(cat_to_name, file_path, model, True, 5)

    # Make a bar graph
    # Create the axes for the bar graph
    fig.add_axes([.5, .1, .225, .225])

    # Get the range for the probabilities
    y_pos = np.arange(len(classes))

    # Plot as a horizontal bar graph
    plt.barh(y_pos, predictions, align='center', alpha=0.5)
    plt.yticks(y_pos, classes)
    plt.xlabel('probabilities')
    plt.show()

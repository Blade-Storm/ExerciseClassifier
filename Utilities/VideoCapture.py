# Video capture utility. Got the idea (and code) from here:
# https://github.com/arundasan91/Machine-Learning/blob/master/OpenCV/youtube-to-frames.py
import cv2
import numpy
import os




# Create a data folder for the images to go into if it doesnt exist
try:
    if not os.path.exists('./data'):
        os.makedirs('./data')
except OSError:
    print('Error Creating directory of data')


# Get the vieo from a mp4 file
video_capture = cv2.VideoCapture('./data/video.mp4')

# Keep track of the frames for each folder (squat, bench, etc...) to prevent duplicate names and overriting files
# Also gives us a breakpoint to exit the loop
currentFrame = 3700

# Loop through the video and save the frames
while(True):
    # Capture the next frame
    ret, frame = video_capture.read()

    # If there are no more frames to capture break out of the loop
    if frame is None:
        break

    # Save the image of the current frame
    name = './data/image' + str(currentFrame) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame)
    #cv2.imshow('frame', frame)

    # Increase current frame for a unique name
    currentFrame +=1
    

# Release the capture and kill any cv windows
video_capture.release()
cv2.destroyAllWindows()



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

#s.system("youtube-dl -o " + 'data/video.mp4' + " -f mp4 " + 'data/video')
# Get the vieo from a url
video_capture = cv2.VideoCapture('./data/video.mp4')

currentFrame = 200
while(True):
    # Capture the next frame
    ret, frame = video_capture.read()

    # Save the image of the current frame
    name = './data/image' + str(currentFrame) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame)
    #cv2.imshow('frame', frame)


    # Increase for a unique name
    currentFrame +=1

    if currentFrame == 250:
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()



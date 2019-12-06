#####################################################################

# Author: Dion Hopkinson-Sibley
# Uses source code by Toby Breckon, toby.breckon@durham.ac.uk
# and StackOverflow (attribution in hist_match.py)

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
import yolo
from hist_match import hist_match

# where is the data ? - set this to where you have it

path_to_dataset = "/home/dion/TTBB-durham-02-10-17-sub10" # ** need to edit this **

# output directory for images, leave empty to not save images
output_dir = ''

# image classes to detect--people and vehicles
allowed_classes = [0,1,2,3,5,6,7,9,11]

# camera data
camera_focal_length = 400
camera_baseline = 0.21

directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = "" # set to timestamp to skip forward to

crop_disparity = True # display full or cropped disparity image
pause_playback = False # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(path_to_dataset, directory_to_cycle_left)
full_path_directory_right =  os.path.join(path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left))

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

max_disparity = 128
stereo_matcher = cv2.StereoSGBM_create(0, max_disparity, 15)

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)


        # Perform histogram matching from the right to the left image
        grayR = hist_match(grayR, grayL).astype(np.uint8)

        # Compute the disparity
        raw_disparity = stereo_matcher.compute(grayL,grayR)
        # Apply median blur to the disparity map
        raw_disparity = cv2.medianBlur(raw_disparity, 5)

        # Scale and threshold the disparity to be within -1 and max_disparity
        disparity = (raw_disparity / 16.)
        _, disparity = cv2.threshold(disparity,max_disparity, max_disparity, cv2.THRESH_TRUNC)

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        _, disparity_display = cv2.threshold(disparity, 0, max_disparity, cv2.THRESH_TOZERO)
        disparity_display = (disparity_display * (256. / max_disparity)).astype(np.uint8)

        # Cover up the car bonnet to avoid detection
        pts = np.array([[0, 544], [0, 535], [440, 395], [650, 395], [1024, 490], [1024, 544]], np.int32)
        pts = pts.reshape((-1,1,2))
        imgL_detect = imgL.copy()
        cv2.fillConvexPoly(imgL_detect, pts, (0,0,0))
        # Cover up car bonnet in display version of disparity map
        pts2 = np.array([[-55, 544], [-55, 515], [385, 388], [630, 388], [1024, 475], [1024, 544]], np.int32)
        pts2 = pts2.reshape((-1,1,2))
        cv2.fillConvexPoly(disparity_display, pts2, (0,0,0))

        windowName = 'left image'
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, imgL.shape[1], imgL.shape[0])

        # Perform YOLO detection
        classIDs, confidences, boxes = yolo.detect(imgL_detect, allowed_classes)
        # Sort objects according to disparity
        for i in range(0, len(boxes)):
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            roi = disparity[max(top,0):top+height, max(left,0):left+width]
            disp = np.mean(roi[roi>=0])
            box.append(disp)
            box.append(classIDs[i])

        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        min_depth = 0
        closest_obj = None

        # Calculate distance to each object.
        # In distance order, calculate the median disparity and then set the
        # values to -1 so that object obscured by others don't use the
        # disparity values belonging to another object
        for i in range(0, len(boxes)):
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            roi = disparity[max(top,0):top+height, max(left,0):left+width]
            depth = camera_focal_length*camera_baseline/(np.median(roi[roi>=0]))
            disparity[
                max(top+height//16,0):top+(14*height)//16,
                max(left+width//16,0):left+(14*width)//16
            ] = -1
            if np.isnan(depth) or depth <= 0:
                depth = -1
            elif depth < min_depth or closest_obj is None:
                closest_obj = yolo.classes[classIDs[i]]
                min_depth = depth
            box.append(depth)

        # Draw the bounding boxes and labels
        for i in range(len(boxes)-1, -1, -1):
            box = boxes[i]
            print(box[6])
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            depth = box[6]
            yolo.drawPred(imgL, yolo.classes[box[5]], depth, left, top, left + width, top + height, (255, 178, 50))

        cv2.imshow(windowName, imgL)

        # crop disparity to chop out left part where there are with no disparity
        # as this area is not seen by both cameras
        #cv2.imshow('d', (disparity * (256. / max_disparity)).astype(np.uint8))

        if (crop_disparity):
            width = np.size(disparity, 1)
            disparity_display = disparity_display[0:,160:width]

        cv2.imshow('Disparity', disparity_display)

        print(filename_left)
        print('%s : %s (%.1fm)'%(filename_right, closest_obj, min_depth))

        # If specified write the prediction and disparity images
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, 'left', filename_left), imgL)
            cv2.imwrite(os.path.join(output_dir, 'right', filename_right), disparity_display)

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        key = cv2.waitKey(400 * (not(pause_playback))) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
            print("-- files skipped (perhaps one is missing or not PNG)")
            print()

# close all windows

cv2.destroyAllWindows()

#####################################################################

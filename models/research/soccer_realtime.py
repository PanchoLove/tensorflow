from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sys
from tkinter import *
import tkinter.filedialog as fdialog

from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Note: Model used for SSDLite_Mobilenet_v2
#PATH_TO_CKPT = './object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_CKPT = './object_detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#print(categories)
#print(category_index)

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()


def detect_team(image, col1, col2, col_gk, show = False):
    # convert to HSV colorbase
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # define color intervals (in HSV colorbase)
    lower_yellow = np.array([25,100,100])
    upper_yellow = np.array([35,255,255])
    lower_lightblue = np.array([95,80,80])
    upper_lightblue = np.array([120,170,255])
    lower_blue = np.array([100,80,80])
    upper_blue = np.array([120,255,255])
    lower_red = np.array([165,50,100])
    upper_red = np.array([180,255,255])
    #lower_red2 = np.array([0,50,100])
    #upper_red2 = np.array([10,255,255])
    lower_purple = np.array([130,80,80])
    upper_purple = np.array([160,255,255])
    lower_green = np.array([35,80,80])
    upper_green = np.array([50,255,255])
    lower_orange = np.array([28,80,80])
    upper_orange = np.array([30,255,255])
    lower_white = np.array([0,0,240])
    upper_white = np.array([190,60,255])
    
    # define the list of boundaries
    # Team 1
    if col1 == 'red':
        rgb1_low = lower_red
        rgb1_up = upper_red
    elif col1 == 'lightblue':
        rgb1_low = lower_lightblue
        rgb1_up = upper_lightblue
    elif col1 == 'yellow':
        rgb1_low = lower_yellow
        rgb1_up = upper_yellow
    elif col1 == 'blue':
        rgb1_low = lower_blue
        rgb1_up = upper_blue
    elif col1 == 'purple':
        rgb1_low = lower_purple
        rgb1_up = upper_purple
    elif col1 == 'green':
        rgb1_low = lower_green
        rgb1_up = upper_green
    elif col1 == 'orange':
        rgb1_low = lower_orange
        rgb1_up = upper_orange
    elif col1 == 'white':
        rgb1_low = lower_white
        rgb1_up = upper_white
    # Team 2
    if col2 == 'red':
        rgb2_low = lower_red
        rgb2_up = upper_red
    elif col2 == 'lightblue':
        rgb2_low = lower_lightblue
        rgb2_up = upper_lightblue
    elif col2 == 'yellow':
        rgb2_low = lower_yellow
        rgb2_up = upper_yellow
    elif col2 == 'blue':
        rgb2_low = lower_blue
        rgb2_up = upper_blue
    elif col2 == 'purple':
        rgb2_low = lower_purple
        rgb2_up = upper_purple
    elif col2 == 'green':
        rgb2_low = lower_green
        rgb2_up = upper_green
    elif col2 == 'orange':
        rgb2_low = lower_orange
        rgb2_up = upper_orange
    elif col2 == 'white':
        rgb2_low = lower_white
        rgb2_up = upper_white
    # Goal-keeper
    if col_gk == 'red':
        rgbGK_low = lower_red
        rgbGK_up = upper_red
    elif col_gk == 'lightblue':
        rgbGK_low = lower_lightblue
        rgbGK_up = upper_lightblue
    elif col_gk == 'yellow':
        rgbGK_low = lower_yellow
        rgbGK_up = upper_yellow
    elif col_gk == 'blue':
        rgbGK_low = lower_blue
        rgbGK_up = upper_blue
    elif col_gk == 'purple':
        rgbGK_low = lower_purple
        rgbGK_up = upper_purple
    elif col_gk == 'green':
        rgbGK_low = lower_green
        rgbGK_up = upper_green
    elif col_gk == 'orange':
        rgbGK_low = lower_orange
        rgbGK_up = upper_orange
    elif col_gk == 'white':
        rgbGK_low = lower_white
        rgbGK_up = upper_white

    boundaries = [
    (rgb1_low, rgb1_up), #red
    (rgb2_low, rgb2_up), #light-blue
    (rgbGK_low, rgbGK_up)
    ]
    # ([25, 146, 190], [96, 174, 250]) #yellow
    i = 0
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(img_hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix/tot_pix
#         print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            return 'Team1' #'red'
        elif ratio > 0.01 and i == 1:
            return 'Team2' #'yellow'
        elif ratio > 0.01 and i == 2:
            return 'GK'

        i += 1
        
        if show == True:
            cv2.imshow("images", np.hstack([image, output]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
              cv2.destroyAllWindows() 
    return 'not_sure'


## To View Color Mask
# filename = 'frame74.jpg' #'image2.jpg'
# image = cv2.imread(filename)
# resize = cv2.resize(image, (640,360))
# detect_team(resize, show=True)


# Para abrir ventana elegir archivo de video
root = Tk()
root.filename =  fdialog.askopenfile(initialdir = "/",title = "Select file",filetypes = (("avi files","*.avi"),("all files","*.*")))
print('Example picture: ', root.filename.name)
example_image_path = root.filename.name
pathAux = example_image_path.find('/',-20)
example_image = example_image_path[pathAux+1:]
print(example_image)
root.destroy()

color_1 = str(input('Color team 1 (red,yellow,blue,lightblue,green,white,etc) : ')) 
color_2 = str(input('Color team 2: ')) 
color_gk = str(input('Color Goalkeeper: ')) 
name_team1 = str(input('Name team 1: ')) 
name_team2 = str(input('Name team 2: ')) 

#intializing the web camera device

#filename = 'DORSALES/D13.avi' #'soccer_small.mp4'
filename = example_image
cap = cv2.VideoCapture(filename)
size = (int(cap.get(3)),
        int(cap.get(4)))

out = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
success = out.open('soccer_out.avi',fourcc,20.0,size,True)

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   counter = 0
   while (True):
      ret, image_np = cap.read()
      counter += 1
      if ret:
          h = image_np.shape[0]
          w = image_np.shape[1]

      if not ret:
        break
      if counter % 1 == 0:
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3,
              min_score_thresh=0.6)
        
          frame_number = counter
          loc = {}
          for n in range(len(scores[0])):
             if scores[0][n] > 0.60:
                # Calculate position
                ymin = int(boxes[0][n][0] * h)
                xmin = int(boxes[0][n][1] * w)
                ymax = int(boxes[0][n][2] * h)
                xmax = int(boxes[0][n][3] * w)

                # Find label corresponding to that class
                for cat in categories:
                    if cat['id'] == classes[0][n]:
                        label = cat['name']

                ## extract every person
                if label == 'person':
                    #crop them
                    crop_img = image_np[ymin:ymax, xmin:xmax]
                    color = detect_team(crop_img,color_1,color_2,color_gk)
                    if color != 'not_sure':
                        coords = (xmin, ymin)
                        if color == 'Team1':
                            loc[coords] = name_team1
                        elif color == 'Team2':
                            loc[coords] = name_team2
                        elif color == 'GK':
                            loc[coords] = 'GK'
                        else:
                            loc[coords] = name_team2
                        
        ## print color next to the person
          for key in loc.keys():
            text_pos = str(loc[key])
            cv2.putText(image_np, text_pos, (key[0], key[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2) # Text in black
      
      print(counter) #cv2.imshow('image', image_np)
      out.write(image_np)
       
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

print('Done!')
cap.release()
out.release()
cv2.destroyAllWindows()
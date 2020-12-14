import os
from openslide import open_slide
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix, recall_score

import tensorflow as tf

class heatmap_on_test():
    
  def __init__(self,level,slide_name,slice_size,model,class_names):
    self.slides_directory_path = '/content/gdrive/My Drive/slides/'
    self.slide_name = slide_name
    self.level = level
    self.slice_size = slice_size
    self.model = model
    self.class_names = class_names

  def read_slide(self, slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

  def get_slide_image(self):
    slide_path = self.slides_directory_path + self.slide_name + '.tif'
    slide = open_slide(slide_path)
    slide_image = self.read_slide(slide, 
                         x=0,
                         y=0,
                         level=self.level,
                         width=slide.level_dimensions[self.level][0],
                         height=slide.level_dimensions[self.level][1])
    return slide_image

  def get_mask_image(self):
    mask_path = self.slides_directory_path + self.slide_name + '_mask.tif'
    mask = open_slide(mask_path)
    mask_image = self.read_slide(mask, 
                         x=0,
                         y=0,
                         level=self.level,
                         width=mask.level_dimensions[self.level][0],
                         height=mask.level_dimensions[self.level][1])[:,:,0]
    return mask_image

  def is_cancerous(self,mask_slice):
    if np.sum(mask_slice) > 0:
      return True
    else:
      return False

  def slice_and_predict(self,test_slide_image,test_slide_actual_mask_image):
    slice_size = self.slice_size
    model = self.model
    n_y = test_slide_image.shape[0]//slice_size
    n_x = test_slide_image.shape[1]//slice_size
    heatmap = np.zeros((test_slide_image.shape[0],test_slide_image.shape[1]))
    actual_class_list = []
    pred_class_list = []
    for i in range(n_y):
      for j in range(n_x):
        mask_slice = test_slide_actual_mask_image[i*slice_size:(i+1)*slice_size,j*slice_size:(j+1)*slice_size]
        actual_class = 1 if self.is_cancerous(mask_slice) else 0
        actual_class_list.append(actual_class)

        slide_slice = test_slide_image[i*slice_size:(i+1)*slice_size,j*slice_size:(j+1)*slice_size,:]
        slide_slice = tf.expand_dims(slide_slice, 0)
        prediction = model.predict(slide_slice)
        score = tf.nn.softmax(prediction[0])
        prediction_class = np.argmax(score)
        pred_class_list.append(prediction_class)

        heatmap[i*slice_size:(i+1)*slice_size,j*slice_size:(j+1)*slice_size] = 1 if class_names[prediction_class] == '1positive' else 0

    return heatmap, actual_class_list, pred_class_list

  def generate(self):
    test_slide_image = self.get_slide_image()
    test_slide_actual_mask_image = self.get_mask_image()
    heatmap, labels, predictions = self.slice_and_predict(test_slide_image,test_slide_actual_mask_image)
    cnf_matrix = confusion_matrix(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    plt.figure(figsize=(10,10), dpi=100)
    plt.imshow(test_slide_image)
    plt.imshow(test_slide_actual_mask_image, cmap='jet', alpha=0.5) # Red regions show actual cancer cells.
    plt.imshow(heatmap, cmap='Greens', alpha=0.5) # Green regions show predicted cancer cells.

    return cnf_matrix, precision, recall


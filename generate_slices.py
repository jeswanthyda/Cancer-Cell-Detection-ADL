import os
from openslide import open_slide
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import pandas as pd

class Slide2Slice():
    
  def __init__(self, slide_names : list, slides_directory_path : str, sliced_slides_dir : str, level : int, slice_size : int = 60):
    self.slide_names = slide_names
    self.slides_directory_path = slides_directory_path
    self.level = level
    self.slice_size = slice_size
    
    if not os.path.exists(sliced_slides_dir):
      os.makedirs(sliced_slides_dir)

    self.level_dir = sliced_slides_dir + f'level{self.level}/'
    if os.path.exists(self.level_dir):
      print(f'Reset level {self.level} directory')
      ! rm -r '$self.level_dir'
    os.makedirs(self.level_dir)
    
  def read_slide(self, slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

  def get_slide_image(self,slide_name):
    slide_path = self.slides_directory_path + slide_name + '.tif'
    slide = open_slide(slide_path)
    slide_image = self.read_slide(slide, 
                         x=0,
                         y=0,
                         level=self.level,
                         width=slide.level_dimensions[self.level][0],
                         height=slide.level_dimensions[self.level][1])
    return slide_image

  def get_mask_image(self,slide_name):
    mask_path = self.slides_directory_path + slide_name + '_mask.tif'
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

  def is_tissue(self,slide_slice, intensity=0.8, tissue_content_threshold = 0.1):
    slide_slice_gray = rgb2gray(slide_slice)
    indices = np.where(slide_slice_gray <= intensity)
    tissue_pixels = list(zip(indices[0], indices[1]))
    tissue_content = len(tissue_pixels) / float(slide_slice_gray.shape[0] * slide_slice_gray.shape[1])
    if tissue_content >= tissue_content_threshold:
      return True
    else:
      return False

  def slice_all_slides(self):
    insight_df = pd.DataFrame(columns=['slide_name','level','negative','positive','negative_with_tissue','positive_with_tissue'])
    self.positive_dir = self.level_dir + '1positive/'
    if not os.path.exists(self.positive_dir):
      os.makedirs(self.positive_dir)

    self.negative_dir = self.level_dir + '0negative/'
    if not os.path.exists(self.negative_dir):
      os.makedirs(self.negative_dir)

    for slide_name in self.slide_names:
      slide_image = self.get_slide_image(slide_name)
      mask_image = self.get_mask_image(slide_name)
      print(f'slicing {slide_name} on level {self.level} and size {self.slice_size}')
      slice_stats = self.slice_one_and_save_to_disk(slide_image,mask_image,slide_name)
      insight_df = insight_df.append(slice_stats,ignore_index=True)
    return insight_df

  def slice_one_and_save_to_disk(self, slide_image, mask_image, slide_name):
    slice_size = self.slice_size
    n_y = slide_image.shape[0]//slice_size
    n_x = slide_image.shape[1]//slice_size
    positive_count = 0
    negative_count = 0
    positive_with_tissue_count = 0
    negative_with_tissue_count = 0
    for i in range(n_y):
      for j in range(n_x):
        im_name = f'{slide_name}_lvl{self.level}_x{j*slice_size}to{(j+1)*slice_size}_y{i*slice_size}to{(i+1)*slice_size}.jpeg'
        slide_slice = slide_image[i*slice_size:(i+1)*slice_size,j*slice_size:(j+1)*slice_size,:]
        mask_slice = mask_image[i*slice_size:(i+1)*slice_size,j*slice_size:(j+1)*slice_size]

        if self.is_cancerous(mask_slice):
          #Save to Positive dir
          im = Image.fromarray(slide_slice)
          im.save(self.positive_dir+im_name)
          positive_count += 1
          if self.is_tissue(slide_slice):
            positive_with_tissue_count += 1
        else:
          #Save to Negative dir
          negative_count += 1
          if self.is_tissue(slide_slice):
            im = Image.fromarray(slide_slice)
            im.save(self.negative_dir+im_name)
            negative_with_tissue_count += 1

    slide_stats = {
        'slide_name' : slide_name,
        'level' : self.level,
        'negative' : negative_count,
        'positive' : positive_count,
        'negative_with_tissue' : negative_with_tissue_count,
        'positive_with_tissue' : positive_with_tissue_count, 
    }
    print(slide_stats)
    return slide_stats

  def run(self):
    insight_df = self.slice_all_slides()
    return insight_df

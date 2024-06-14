#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import cv2
from tensorpack.dataflow import imgaug
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import random
from PIL import Image, ImageOps

class RandomCropLetterbox(imgaug.ImageAugmentor):

  def __init__(self, output_w, output_h, jitter, fill_color=127):
      self.output_w = output_w
      self.output_h = output_h
      self.jitter = jitter
      self.fill_color = 127
      self.crop_info = None
      self._init(locals())

  def _get_augment_params(self, img):
      return None

  def _augment(self, img_np, params):
      '''
      @param: img_np (ndarray) (width, height, channel)
      '''
      #print(img_np.dtype, img_np.shape)
      img = Image.fromarray(img_np)
      orig_w, orig_h = img.size
      channels = img_np.shape[2]
      dw = int(self.jitter * orig_w)
      dh = int(self.jitter * orig_h)
      new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
      scale = random.random()*(2-0.25) + 0.25
      if new_ar < 1:
         nh = int(scale * orig_h)
         nw = int(nh * new_ar)
      else:
         nw = int(scale * orig_w)
         nh = int(nw / new_ar)

      if self.output_w > nw:
         dx = random.randint(0, self.output_w - nw)
      else:
         dx = random.randint(self.output_w - nw, 0)

      if self.output_h > nh:
         dy = random.randint(0, self.output_h - nh)
      else:
         dy = random.randint(self.output_h - nh, 0)

      nxmin = max(0, -dx)
      nymin = max(0, -dy)
      nxmax = min(nw, -dx + self.output_w - 1)
      nymax = min(nh, -dy + self.output_h - 1)
      sx, sy = float(orig_w)/nw, float(orig_h)/nh
      orig_xmin = int(nxmin * sx)
      orig_ymin = int(nymin * sy)
      orig_xmax = int(nxmax * sx)
      orig_ymax = int(nymax * sy)
      orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
      orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
      output_img = Image.new(img.mode, (self.output_w, self.output_h), color=(self.fill_color,)*channels)
      output_img.paste(orig_crop_resize, (0, 0))
      self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]

      img_np = np.array(output_img)
      return img_np


  def _augment_coords(self, coords, param):
      annos = coords.reshape(-1, 4) #restore the layout (x1, y1, x2, y2)
      sx, sy, crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.crop_info
      #print(self.crop_info)

      annos[:,0] = np.maximum(crop_xmin, np.round(annos[:,0]/sx))
      annos[:,2] = np.minimum(crop_xmax, np.round(annos[:,2]/sx))
      annos[:,1] = np.maximum(crop_ymin, np.round(annos[:,1]/sy))
      annos[:,3] = np.minimum(crop_ymax, np.round(annos[:,3]/sy))
      annos[:, [0, 2]] -= crop_xmin
      annos[:, [1, 3]] -= crop_ymin
      coords = annos.reshape(-1,2)

      return coords

class RandomFlip(imgaug.ImageAugmentor):

  def __init__(self, threshold):
      self.threshold = threshold
      self._init(locals())
      self.flip = False
      self.im_w = None


  def _get_augment_params(self, img):
      return None

  def _augment(self, img_np, params):
      img = Image.fromarray(img_np)
      #print(img_np.shape)
      self._get_flip()
      self.im_w = img.size[0]
      if self.flip:
         img = img.transpose(Image.FLIP_LEFT_RIGHT)
      img_np = np.array(img)
      #print(img_np.shape)
      return img_np

  def _get_flip(self):
      self.flip = random.random() < self.threshold

  def _augment_coords(self, coords, param):
      annos = coords.reshape(-1, 4) #restore the (x1, y1, x2, y2) layout
      width = annos[:,2] - annos[:,0]

      if self.flip and self.im_w is not None:
         annos[:,0] = self.im_w - annos[:,0] - width
         annos[:,2] = annos[:,0] + width

      coords = annos.reshape(-1, 2)
      return coords

class HSVShift(imgaug.ImageAugmentor):

  def __init__(self, hue, saturation, value):
      self.hue = hue
      self.saturation = saturation
      self.value = value
      self._init(locals())


  def _augment(self, img_np, params):
      dh = random.uniform(-self.hue, self.hue)
      ds = random.uniform(1, self.saturation)
      if random.random() < 0.5:
         ds = 1/ds
      dv = random.uniform(1, self.value)
      if random.random() < 0.5:
         dv = 1/dv
      img = Image.fromarray(img_np)
      img = img.convert('HSV')
      channels = list(img.split())

      def change_hue(x):
          x += int(dh * 255)
          if x > 255:
              x -= 255
          elif x < 0:
              x += 255
          return x

      channels[0] = channels[0].point(change_hue)
      channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*ds))))
      channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*dv))))

      img = Image.merge(img.mode, tuple(channels))
      img = img.convert('RGB')

      img_np = np.array(img)
      return img_np

class Letterbox(imgaug.ImageAugmentor):
  def __init__(self, dimension=None):
      self.dimension = dimension
      self.pad = None
      self.scale = None
      self.fill_color = 127
      self._init(locals())


  def _augment(self, img_np, params):
      net_w = self.dimension
      net_h = self.dimension
      img = Image.fromarray(img_np)
      im_w, im_h = img.size

      if im_w == net_w and im_h == net_h:
         self.scale = None
         self.pad = None
         return np.array(img)

      # Rescaling
      if im_w / net_w >= im_h / net_h:
          self.scale = net_w / im_w
      else:
          self.scale = net_h / im_h
      if self.scale != 1:
          resample_mode = Image.NEAREST #Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
          img = img.resize((int(self.scale*im_w), int(self.scale*im_h)), resample_mode)
          im_w, im_h = img.size

      if im_w == net_w and im_h == net_h:
          self.pad = None
          return np.array(img)

      # Padding
      img_np = np.array(img)
      channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
      pad_w = (net_w - im_w) / 2
      pad_h = (net_h - im_h) / 2
      self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
      img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,)*channels)

      img_np = np.array(img)
      return img_np

  def _get_augment_params(self, img):
      return None

  def _augment_coords(self, coords, param):
      annos = coords.reshape(-1, 4) #restore the (x1, y1, x2, y2) layout
      #print(self.scale, self.pad)
      for anno in annos:
          if self.scale is not None:
             anno *= self.scale
          if self.pad is not None:
             anno[0] += self.pad[0]
             anno[1] += self.pad[1]
             anno[2] += self.pad[0]
             anno[3] += self.pad[1]
      coords = annos.reshape(-1, 2)
      return coords

def show_im(img, bbox):
    plt.clf()
    ax = plt.subplot(1,1,1)
    img = img[:,:,::-1]
    ax.imshow(img.squeeze())
    for box in bbox:
      rect = patches.Rectangle((box[0], box[1]),
                                box[2]-box[0], box[3] - box[1],
                                edgecolor='r',facecolor='none')
      ax.add_patch(rect)
    plt.show()
    plt.close()

def train_img_augment(img, bbox):
    '''
    img: (width, height, 3) numpy ndarray
    bbox: Nx4 numpy ndarray (x11, y11, x12, y12  # x-y coordinate of the topleft/rightdown points of the first bbox
                             x21, y21, x22, y22
                             x31, y31, x32, y32
                             ..................
                            )

    '''
    #aug = imgaug.AugmentorList([imgaug.Grayscale(), imgaug.Affine(scale=(0.8, 1.), translate_frac=(0.01, 0.03), rotate_max_deg=10)])
    aug = imgaug.AugmentorList([HSVShift(0.1, 1.5, 1.5), RandomFlip(0.5), RandomCropLetterbox(416, 416, 0.2, 127)])
    new_img, param = aug.augment_return_params(img)
    new_coord = aug.augment_coords(bbox.reshape(-1, 2), param)
    new_coord = new_coord.reshape(-1, 4)
    return new_img, new_coord

def validate_img_adjust(img, bbox):
    aug = imgaug.AugmentorList([Letterbox(416)])
    new_img, param = aug.augment_return_params(img)
    new_coord = aug.augment_coords(bbox.reshape(-1, 2), param)
    new_coord = new_coord.reshape(-1, 4)
    return new_img, new_coord

def test_img_adjust(img):
    aug = imgaug.AugmentorList([Letterbox(416)])
    new_img, _ = aug.augment_return_params(img)
    return new_img



argparser = argparse.ArgumentParser(
    description='Data augmentation for YOLO_v2.')

argparser.add_argument(
    '-f',
    '--file',
    help='path to demo image')


if __name__ == '__main__':
   args = argparser.parse_args()
   im_path = args.file
   im = cv2.imread(im_path)
   coords = np.array([[48., 240., 195., 371.], [8., 12., 352., 498.]])
   print("Coordinates before augmentation:{}".format(coords))
   show_im(im, coords)
   new_im, new_cord = train_img_augment(im, coords)
   #new_im, new_cord = validate_img_adjust(im, coords)
   print("Coordinates/Shape after augmentation:{}/{}".format(new_cord, new_im.shape))
   show_im(new_im, new_cord)

import sys
sys.path.append('/path/to/caffe/python')
import caffe
import io
from PIL import Image
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy


frames_root_folder = '/path/to/data/folder/'


def processImageCrop(im_info, transformer):
  im_path = im_info[0]
  im_crop = im_info[1]
  im_reshape = im_info[2]
  im_flip = im_info[3]
  data_in = caffe.io.load_image(im_path)
  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, im_reshape)
  if im_flip:
    data_in = caffe.io.flip_image(data_in, 1) 
    data_in = data_in[im_crop[0]:im_crop[2], im_crop[1]:im_crop[3], :] 
  processed_image = transformer.preprocess('data_in',data_in)
  return processed_image

class ImageProcessorCrop(object):
  def __init__(self, transformer):
    self.transformer = transformer
  def __call__(self, im_info):
    return processImageCrop(im_info, self.transformer)

class sequenceGeneratorVideo(object):
  def __init__(self, buffer_size, clip_length, path_to_image, num_videos, video_dict, video_order):
    self.buffer_size = buffer_size
    self.clip_length = clip_length
    self.N = self.buffer_size*self.clip_length
    self.path_to_image = path_to_image
    self.num_videos = num_videos
    self.video_dict = video_dict
    self.video_order = video_order
    self.idx = 0

  def __call__(self):
    label_pr = []
    im_paths = []
    im_crop = []
    im_reshape = []  
    im_flip = []

 
    if self.idx + self.buffer_size >= self.num_videos:
      idx_list = range(self.idx, self.num_videos)
      idx_list.extend(range(0, self.buffer_size-(self.num_videos-self.idx)))
    else:
      idx_list = range(self.idx, self.idx+self.buffer_size)
    idx_count = 1
    idx_total_count = 0
    
    while idx_count !=0 :
      idx_count = 0
      idx_list_last = idx_list[-1]
      delete_idxs = []      
      for k in range(self.buffer_size) :        
        idx_num = idx_list[k]
        key = self.video_order[idx_num]
        frame_num = int(key.split('-')[1].split('.')[0])
        if frame_num <= self.clip_length:
          delete_idxs.append(k)
          idx_count = idx_count + 1
          idx_total_count = idx_total_count + 1
      for delete_idx in sorted(delete_idxs, reverse = True) :
        del idx_list[delete_idx]
      for h in range(idx_count):
        idx_list.append(idx_list_last+h+1)


    for i in idx_list:
      key = self.video_order[i]
      frame_num = int(key.split('-')[1].split('.')[0])    
      video_num = key.split('-')[0].split('o')[1]
      frames = []

      for j in range(self.clip_length):
        keynew = 'video' + video_num + '-' + str(frame_num-j) + '.jpg'
        keynew_fullpath = self.path_to_image + 'video' + video_num + '/' + keynew
        frames.append(keynew_fullpath)
        label_phase = self.video_dict[keynew]['label']
        label_pr.append(label_phase)

      im_paths.extend(frames[::-1])
      print im_paths
      video_reshape = self.video_dict[key]['reshape']
      video_crop = self.video_dict[key]['crop']
      im_reshape.extend([(video_reshape)]*self.clip_length)
      r0 = int(random.random()*(video_reshape[0] - video_crop[0]))
      r1 = int(random.random()*(video_reshape[1] - video_crop[1]))
      im_crop.extend([(r0, r1, r0+video_crop[0], r1+video_crop[1])]*self.clip_length)     
      f = random.randint(0,1)  
      im_flip.extend([f]*self.clip_length)
    
    im_info = zip(im_paths,im_crop, im_reshape, im_flip)

    self.idx = self.idx + self.buffer_size + idx_total_count
    if self.idx >= self.num_videos:
      self.idx = self.idx - self.num_videos

    return label_pr, im_info
  
def advance_batch(result, sequence_generator, image_processor, pool):
  
    label_pr, im_info = sequence_generator()
    tmp = image_processor(im_info[0])
    result['data'] = pool.map(image_processor, im_info)
    result['label'] = label_pr 
    cm = np.ones(len(label_pr))
    cm[0::3] = 0
    result['clip_markers'] = cm

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)

class videoInput(caffe.Layer):

  def initialize(self):
    self.train_or_test = 'test'
    self.buffer_size = 2  #num videos processed per batch
    self.frames = 3   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 224
    self.width = 224
    self.path_to_images = frames_root_folder 
    self.video_list = '/path/to/test/gt_file/gt_test_shuffle.txt' 

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.video_list, 'r')
    f_lines = f.readlines()
    f.close()

    video_dict = {} 
    current_line = 0
    self.video_order = [] 
    for ix, line in enumerate(f_lines):
      video_frame = line.split(' ')[0].split('/')[1]
      label_phase = int(line.split(' ')[1])

      video_dict[video_frame] = {} 
      video_dict[video_frame]['reshape'] = (250,250)
      video_dict[video_frame]['crop'] = (224, 224)
      video_dict[video_frame]['label'] = label_phase
      self.video_order.append(video_frame) 

    self.video_dict = video_dict
    self.num_videos = len(video_dict.keys())

    shape = (self.N, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    image_mean = [75, 75, 101]   #change this to our image mean
    channel_mean = np.zeros((3,224,224))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = 24

    self.image_processor = ImageProcessorCrop(self.transformer)
    self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.frames, self.path_to_images, self.num_videos, self.video_dict, self.video_order)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, self.pool)
    self.dispatch_worker()
    self.top_names = ['data', 'label', 'clip_markers']
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'data':
        shape = (self.N, self.channels, self.height, self.width)
      elif name == 'label':
        shape = (self.N,)
      elif name == 'clip_markers':
        shape = (self.N,)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    new_result_data = [None]*len(self.thread_result['data']) 
    new_result_label_phase = [None]*len(self.thread_result['label'])
    new_result_cm = [None]*len(self.thread_result['clip_markers'])
    for i in range(self.frames):
      for ii in range(self.buffer_size):
        old_idx = ii*self.frames + i
        new_idx = i*self.buffer_size + ii
        new_result_data[new_idx] = self.thread_result['data'][old_idx]
        new_result_label_phase[new_idx] = self.thread_result['label'][old_idx]
        new_result_cm[new_idx] = self.thread_result['clip_markers'][old_idx]

    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'data':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data[i] 
      elif name == 'label':
        top[top_index].data[...] = new_result_label_phase
      elif name == 'clip_markers':
        top[top_index].data[...] = new_result_cm


    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass


class TrainSV(videoInput):   

  def initialize(self):
    self.train_or_test = 'train'
    self.buffer_size = 10  #num videos processed per batch
    self.frames = 3   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 224
    self.width = 224
    self.path_to_images = frames_root_folder 
    self.video_list = '/path/to/train/gt_file/gt_train_shuffle.txt' 

class TestSV(videoInput):   

  def initialize(self):
    self.train_or_test = 'test'
    self.buffer_size = 2  #num videos processed per batch
    self.frames = 3   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 224
    self.width = 224
    self.path_to_images = frames_root_folder 
    self.video_list = '/path/to/test/gt_file/gt_test_shuffle.txt' 

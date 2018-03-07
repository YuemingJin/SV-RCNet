from __future__ import division
import numpy as np
import glob
caffe_root = '/path/to/caffe/' #need to modify!!!!!
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
import pickle
import os
import codecs
import time




def initialize_transformer(image_mean, is_flow):
  shape = (10*3, 3, 224, 224)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,224,224))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val  #make each channel value(channel_mean 227*227) equal to each channal(image_mean 1)
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', is_flow)
  return transformer
  #change the caffe transformer

def SVRCNet_test_video(frames, net, transformer, is_flow):
  clip_length = 3   ## length of input video clip
  step = 1          ## stride number. 1 for Cholec80; 3 for M2CAI
  downsample = 1    ## downsample rate in testing process. set 1 for Cholec80 since we have downsampled the data at the preprocessing. set 5 for M2CAI
  internal = step * downsample 
  begin = step * downsample * (clip_length - 1) 
  input_images = []
  j = 0
  for im in frames:
    input_im = caffe.io.load_image(im)
    if (input_im.shape[0] > 250):
      print 'resizing image......',j
      input_im = caffe.io.resize_image(input_im, (250,250))
      j = j + 1
    input_images.append(input_im) 
  vid_length = len(input_images)
  output_predictions = np.zeros((vid_length,7)) 
  
  for i in range(begin,vid_length):
    input_data = []
    print 'adding image.......',i
    h = clip_length #3
    for j in range(clip_length):
      h = h - 1
      index = i - internal * h
      input_data.append(input_images[index])
      

    print 'testing...',i
    clip_input = input_data
    clip_input = caffe.io.oversample(clip_input,[224,224])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0

    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)

    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data',inputs)

    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))

    current_clip_pred = np.mean(out['probs'],1)
    #current_pred = current_clip_pred[clip_length-1] ### last one
    current_pred = np.mean(current_clip_pred,0)  ### mean

    if i == begin:
      for k in range(begin+1):
        output_predictions[k] = current_pred
    else:
      output_predictions[i] = current_pred

  return output_predictions




def each_video(results_folder,root_folder,video):


  caffe.set_device(1) #gpu device number
  caffe.set_mode_gpu()

  ucf_mean_RGB = np.zeros((3,1,1))
  ucf_mean_RGB[0,:,:] = 75
  ucf_mean_RGB[1,:,:] = 75
  ucf_mean_RGB[2,:,:] = 101
  # 75,75,101 are the mean values for three channels

  transformer_RGB = initialize_transformer(ucf_mean_RGB, False)

  #Models and weights
  svrcnet_model = '/path/to/deploy/folder/SVRCNet-workflow-deploy.prototxt'  
  svrcnet_caffemodel = '/path/to/caffemodel/folder/name.caffemodel'
  svrcnet_net =  caffe.Net(svrcnet_model, svrcnet_caffemodel, caffe.TEST)

  # create files to save results
  filename = results_folder + video +"_phase.txt"
  fw = codecs.open(filename, "w", "utf-8-sig")

  filename_pre = results_folder + video +"_phase_prediction.txt"
  fw_pre = codecs.open(filename_pre, "w", "utf-8-sig")


  # load test data path
  frames = sorted(glob.glob('%s%s/*.jpg' %(root_folder, video)), key = os.path.getmtime)

  clip_length = 3
  total_num  = len(frames) #frame number of each video
  frames = frames[0:total_num -1]
  frames_num= total_num -1
  print frames_num

  action_hash = pickle.load(open('/path/to/action_hash_rev_cholec80.p','rb'))


  # initialize the model
  predictions_SVRCNet = SVRCNet_test_video(frames, svrcnet_net, transformer_RGB, False)
  del svrcnet_net


  #test
  headlines = ''.join(['Frame', '\t', 'Phase', '\n'])
  fw.writelines(headlines)

  for i in range(frames_num):
    
    frame_prediction = np.zeros((1,8))
    frame_prediction = predictions_SVRCNet[i,:]
    max_label = frame_prediction.argmax()
    top_inds = frame_prediction.argsort()[::-1][:3]
    pre = zip(frame_prediction[top_inds], top_inds)

    index = i * 25
    fileline_v2 = ''.join([str(index),'\t', action_hash[max_label],'\n'])
    fw.writelines(fileline_v2)


    fileline_v2_pre = ''.join([str(i),'\t', str(frame_prediction[0]), '\t', str(frame_prediction[1]), '\t', str(frame_prediction[2]), '\t', str(frame_prediction[3]), '\t', str(frame_prediction[4]), '\t', str(frame_prediction[5]), '\t', str(frame_prediction[6]), '\n'])
    fw_pre.writelines(fileline_v2_pre)


  fw.close()
  fw_pre.close()




def prediction_all():
  results_folder = 'path/to/result/folder/'
  root_folder = '/path/to/resized_data/folder/'
  if os.path.exists(results_folder):
    print 'folder exist'
  else:
    os.makedirs(results_folder)

  for i in range(41,81):
    video = 'video' +str(i)
    print video
    each_video(results_folder,root_folder,video)



if __name__ == '__main__':
  prediction_all()
















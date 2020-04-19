# -*- coding: utf-8 -*-
"""
@author: gxd
"""

import numpy as np
import math
import skimage.measure as ski
from PIL import Image
import tensorflow as tf
import net_final as net
import glob
import os

def cal_PSNR(ori_img, dst_img):
    #assume RGB image / Y image
    dst_data = np.array(dst_img, dtype=np.float64)
    ori_data = np.array(ori_img,dtype=np.float64)
    diff = dst_data - ori_data
    diff = diff.flatten('C')
    mse_sqrt = np.sqrt(np.mean(diff ** 2.))
    if mse_sqrt <= 1e-10:
        return 100
    else:
        return 20*math.log10(255.0/mse_sqrt)
    


def cal_SSIM(true_img, test_img, MultiChannel_Flag):
    #if img has more than 1 channel (like RGB), MultiChannel_Flag need to be 'True'; 
    #else img only has 1 channel (like Y), MultiChannel_Flag need to be 'False'
    true_data = np.array(true_img)
    test_data = np.array(test_img)
    
    return ski.compare_ssim(true_data,test_data,multichannel=MultiChannel_Flag)

   
 
def read_data(rain_video_path, cur_index):
# frame_index the index of the frame which will be derained
    rain_seq = glob.glob(rain_video_path+'/*.jpg')
    rain_seq.sort()
    
    x1_name = rain_seq[cur_index - 2]
    x2_name = rain_seq[cur_index - 1]
    x3_name = rain_seq[cur_index]
    x4_name = rain_seq[cur_index + 1]
    x5_name = rain_seq[cur_index + 2]       
        
    x1_img=Image.open(x1_name)
    x2_img=Image.open(x2_name)
    x3_img=Image.open(x3_name)
    x4_img=Image.open(x4_name)
    x5_img=Image.open(x5_name)
    
    #crop img if the img width or height is not the a multiple of 4
    width=x3_img.size[0]
    height=x3_img.size[1]
    if ((width % 4 != 0) | (height % 4 != 0)):
        crop_box=[0, 0, width-(width % 4), height-(height % 4)]
        x1_img=x1_img.crop(crop_box)
        x2_img=x2_img.crop(crop_box)
        x3_img=x3_img.crop(crop_box)
        x4_img=x4_img.crop(crop_box)
        x5_img=x5_img.crop(crop_box)
                    
                    
    x1_YCbCr=x1_img.convert("YCbCr")
    x2_YCbCr=x2_img.convert("YCbCr")
    x3_YCbCr=x3_img.convert("YCbCr")
    x4_YCbCr=x4_img.convert("YCbCr")
    x5_YCbCr=x5_img.convert("YCbCr")
                
    x1_Y_tmp,_,_=x1_YCbCr.split() 
    x2_Y_tmp,_,_=x2_YCbCr.split()  
    x3_Y_tmp,_,_=x3_YCbCr.split()  
    x4_Y_tmp,_,_=x4_YCbCr.split()  
    x5_Y_tmp,_,_=x5_YCbCr.split() 
    
    hh=x1_Y_tmp.size[1]
    ww=x1_Y_tmp.size[0]
    x1_Y=np.zeros([1, hh, ww, 1])
    x2_Y=np.zeros([1, hh, ww, 1])
    x3_Y=np.zeros([1, hh, ww, 1])
    x4_Y=np.zeros([1, hh, ww, 1])
    x5_Y=np.zeros([1, hh, ww, 1])
    
    x1_Y[0,:,:,0]=np.array(x1_Y_tmp)/255.0
    x2_Y[0,:,:,0]=np.array(x2_Y_tmp)/255.0
    x3_Y[0,:,:,0]=np.array(x3_Y_tmp)/255.0
    x4_Y[0,:,:,0]=np.array(x4_Y_tmp)/255.0
    x5_Y[0,:,:,0]=np.array(x5_Y_tmp)/255.0
    
    
    return x1_Y, x2_Y, x3_Y, x4_Y, x5_Y, x3_YCbCr



def test():
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # select GPU device
    
    tf.reset_default_graph() 
    
    
    #input information of the testing data
    testdata_dir = '../test_data/'
    video_index = 1
    
    if video_index==1: # 1-r1 video whose resolution is 640x480,
        test_WIDTH = 640
        test_HEIGHT = 480
    elif video_index==2: #2-r2 video whose resolution is 528x282
        test_WIDTH = 528
        test_HEIGHT = 282
    elif video_index==3: #3-r2 video whose resolution is 640x640
        test_WIDTH = 640
        test_HEIGHT = 640    
    
    #choose the testing network
    only_DR_subnet_flag=0 #0-use the whole net of EEVRSR consisting of a motion compensation sub-net and a de-raining sub-net; 1-only use the de-raining sub-net of EEVSRSR,    
    
    #output path
    if only_DR_subnet_flag:
        out_path = './EEVRSR_result/DR_subnet_result/' 
    else:
        out_path = './EEVRSR_result/all_net_result/' 
    
    #Session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
        
    #Placeholder
    if ((test_WIDTH % 4 != 0) | (test_HEIGHT % 4 != 0)):
        test_WIDTH -= (test_WIDTH % 4)
        test_HEIGHT -= (test_HEIGHT % 4)
        
    is_train = tf.placeholder(tf.bool, shape=[])        
    X0_test = tf.placeholder(tf.float32, shape=[1, test_HEIGHT, test_WIDTH, 1])
    Xm1_test = tf.placeholder(tf.float32, shape=[1, test_HEIGHT, test_WIDTH, 1])
    Xm2_test = tf.placeholder(tf.float32, shape=[1, test_HEIGHT, test_WIDTH, 1])
    Xp1_test = tf.placeholder(tf.float32, shape=[1, test_HEIGHT, test_WIDTH, 1])
    Xp2_test = tf.placeholder(tf.float32, shape=[1, test_HEIGHT, test_WIDTH,1])
    
    if only_DR_subnet_flag:
        X0_derained_Y=net.test_network_deraining_subnet(is_train, X0_test, Xm1_test, Xm2_test, Xp1_test, Xp2_test, 1) # test only the de-raining sub-net
    else:
        X0_derained_Y=net.test_network_all(is_train, X0_test, Xm1_test, Xm2_test, Xp1_test, Xp2_test, 1)  # test EEVRSR
    
    
    #Load model
    if only_DR_subnet_flag:
    	#load model for testing only the de-raining sub-net
        ref_vars_derain = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DerainNet')
        saver_restore_derain = tf.train.Saver(ref_vars_derain)
        saver_restore_derain.restore(sess, '../final_model/EEVRSR_derain_only/model.ckpt')    
    else:
    	#load model for testing the whole EEVRSR net
        ref_vars_derain = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DerainNet')
        saver_restore_derain = tf.train.Saver(ref_vars_derain)
        saver_restore_derain.restore(sess, '../final_model/EEVRSR_all/model.ckpt')
        
        ref_vars_mc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MCNet')
        saver_restore_mc = tf.train.Saver(ref_vars_mc)
        saver_restore_mc.restore(sess, '../final_model/EEVRSR_all/model.ckpt')   
	 
    #Import data && Run test      
    rain_path=testdata_dir+'r'+str(video_index)
    num_img=len(glob.glob(rain_path+'/*.jpg'))
    out_dir=out_path +'r'+str(video_index)+'_EEVRSR/'
    out_folder = os.path.exists(out_dir)
    if not out_folder:               
        os.makedirs(out_dir)
    
    print("num_img: "+str(num_img))
    for img_index in range(2,num_img-2):
        ##Import data 
        x1_test, x2_test, x3_test, x4_test, x5_test, x3_YCbCr = read_data(rain_path, img_index)
        test_feed_dict = {
                is_train: False, 
                X0_test: x3_test, 
                Xm1_test: x2_test, 
                Xm2_test: x1_test, 
                Xp1_test: x4_test, 
                Xp2_test: x5_test,
        }

        ##Run test
        x0_Y_derained_tmp = sess.run([X0_derained_Y], feed_dict=test_feed_dict)
        
        #yuv2rgb
        x0_Y_derained=np.array(x0_Y_derained_tmp)[0,0,:,:,0] * 255.0
        x0_Y_derained=np.clip(np.round(x0_Y_derained), 0, 255)
        x0_Y_derained=Image.fromarray(x0_Y_derained.astype('uint8'))
        _,x0_U,x0_V=x3_YCbCr.split()
        x0_derained=Image.merge('YCbCr',[x0_Y_derained,x0_U,x0_V])
        derained_img=x0_derained.convert('RGB')
        
        ##save as jpg
        out_name = out_dir + str("%05d" % (img_index+1)) + '.jpg'
        derained_img.save(out_name)
      

if __name__ == '__main__':
    test()
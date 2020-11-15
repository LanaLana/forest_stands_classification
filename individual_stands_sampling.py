import os
import cv2
import sys
import math
import scipy
import random
import rasterio
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import seed
from glob import glob
from rasterio.windows import Window

from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D
from keras import layers
from keras.models import Model
import tensorflow as tf
import random
import json

from augmentation import augmentation 

class Generator:
    def __init__(self, batch_size, class_0, class_1, num_channels):
        self.num_channels = num_channels
        self.num_classes = 2
        self.IMG_ROW = 64
        self.IMG_COL = 64
        self.batch_size = batch_size
        self.class_0 = class_0
        self.class_1 = class_1
        self.cloud = False
        self.augm = False
        self.stands_id = False
        self.per_stand_loss = False
        
        self.intersect_layer = []
        
        self.json_file_linden_val = None
        self.json_file_oak_val = None
        self.json_file_linden_train = None
        self.json_file_oak_train = None
            
    def get_img_mask_array(self, imgpath, upper_left_x, upper_left_y, pol_width, pol_height, crop_id, age_flag = False):
        with rasterio.open(imgpath+'/'+'B02.tif') as src:
            size_x = src.width
            size_y = src.height
        difference_x = max(0, self.IMG_COL - int(pol_width))
        difference_y = max(0, self.IMG_ROW - int(pol_height))
        
        rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),
                               min(size_x, int(upper_left_x) + int(pol_width) + difference_x) - self.IMG_COL)
        rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),
                               min(size_y, int(upper_left_y) + int(pol_height) + difference_y) - self.IMG_ROW)
        
        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
        
        if self.stands_id:
            id_mask = np.zeros((self.IMG_ROW, self.IMG_COL))
            with rasterio.open(imgpath + '/id.tif') as src:
                id_mask = src.read(window=window).astype(np.uint32)
                id_mask = np.where(id_mask == int(crop_id), 1, 0)

        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_0:
            #if '{}.tif'.format(cl_name) in os.listdir(imgpath):
            with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                mask_0 += src.read(window=window).astype(np.uint8)
                
            if len(self.intersect_layer):
                with rasterio.open(imgpath + '/{}.tif'.format(self.intersect_layer[0])) as src:
                    mask_0 *= src.read(window=window).astype(np.uint8)
        
        mask_1 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_1:
            with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                mask_1 += src.read(window=window).astype(np.uint8)
        
            if len(self.intersect_layer):
                layer_ind = len(self.intersect_layer)
                with rasterio.open(imgpath + '/{}.tif'.format(self.intersect_layer[layer_ind-1])) as src:
                    mask_1 *= src.read(window=window).astype(np.uint8)
       
        img = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.float)
        for i, ch in enumerate(['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']):
            with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                img[:,:,i] = src.read(window=window)
        
        img /= 10000.
        img = img.clip(0, 1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # AGE
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if age_flag:
            channel_name = '_age.tif'
            with rasterio.open(imgpath + channel_name) as src:
                img[:,:,-1] = src.read(window=window).astype(np.float)
            img[:,:,-1] = (img[:,:,-1] / 100.).clip(0., 1.)
         
        mask = np.ones((self.IMG_ROW, self.IMG_COL, self.num_classes)) 
        if self.stands_id:
            mask[:,:,0] = ((mask_0 + mask_1)>0.5)*np.where(np.argmax(np.array([mask_0, mask_1]), axis=0)==0, 1, 0)* id_mask #*mask_0
            mask[:,:,1] = ((mask_0 + mask_1)>0.5)*np.where(np.argmax(np.array([mask_0, mask_1]), axis=0)==1, 1, 0)* id_mask #*mask_1
        else:
            mask[:,:,0] = ((mask_0 + mask_1)>0.5)*np.where(np.argmax(np.array([mask_0, mask_1]), axis=0)==0, 1, 0)
            mask[:,:,1] = ((mask_0 + mask_1)>0.5)*np.where(np.argmax(np.array([mask_0, mask_1]), axis=0)==1, 1, 0)
            
        if self.per_stand_loss:
            mask[:,:,0] *= mask_0[0,:,:]
            mask[:,:,1] *= mask_1[0,:,:]
        #---------------------------------------------------------------------------------------
        # augmentation
        #---------------------------------------------------------------------------------------
        if self.augm:
            # color augmentation is implemented just for RGB img
            img, mask_tmp  = augmentation(img, mask, color_aug_prob=0)
            #if len(mask_tmp.shape)==2:
            #    mask[:,:,0]=mask_tmp
            #else:
            mask=mask_tmp
        
        if self.per_stand_loss:
            mask = mask.astype('float64')
            mask[:,:,0] *= 0.1
            mask[:,:,1] *= 0.1
            
        return np.asarray(img), np.asarray(mask) #, np.max([np.max(mask_0), np.max(mask_1)]) 
    
    def extract_val(self, sample):
        return sample['upper_left_x'], sample['upper_left_y'], sample['pol_width'], sample['pol_height']
    
    def train_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                if random.random() > 0.5:
                    random_key = random.choice(list(self.json_file_cl0_train.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_train[random_key])
                else:
                    random_key = random.choice(list(self.json_file_cl1_train.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl1_train[random_key])
                img_name = random_key[:-len(random_key.split('_')[-1])-1]
                img,mask=self.get_img_mask_array(img_name, upper_left_x, upper_left_y, 
                                                 pol_width, pol_height, random_key.split('_')[-1])
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[] 

    def val_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                if random.random() > 0.5:
                    random_key = random.choice(list(self.json_file_cl0_val.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_val[random_key])
                else:
                    random_key = random.choice(list(self.json_file_cl1_val.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl1_val[random_key])
                img_name = random_key[:-len(random_key.split('_')[-1])-1]
                img,mask=self.get_img_mask_array(img_name, upper_left_x, upper_left_y, 
                                                 pol_width, pol_height, random_key.split('_')[-1])
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[]
            
    def set_prob(self):
        img_prob = np.zeros((len(self.train_img_list)))
        for i, img_path in enumerate(self.train_img_list):
            for cl in self.class_0+self.class_1:
                if cl+'.tif' in os.listdir(img_path):
                    img_prob[i] += np.sum(tiff.imread(img_path+'/'+cl+'.tif'))
        img_prob = img_prob/np.sum(img_prob)
        return img_prob

    def weighted_categorical_crossentropy(self, weights):
        def loss(target,output,from_logits=False):
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            non_zero_pixels = tf.reduce_sum(target, axis=-1)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1) \
                    * (self.IMG_ROW*self.IMG_COL*self.batch_size) / K.sum(non_zero_pixels)

        return loss
    
    def read_json(self, folders, class_name):
        js_full = {}
        samples_set = set()
        for folder in folders:
            json_file = '{}/{}.json'.format(folder, class_name)
            with open(json_file, 'r') as f:
                js_tmp = json.load(f)
            keys_list = set(js_tmp.keys())
            for key in keys_list:
                #if tuple(self.extract_val(js_tmp[key])) not in samples_set:
                #    js_tmp[folder+'_'+key] = js_tmp[key]
                #    samples_set.add(tuple(self.extract_val(js_tmp[key])))
                js_tmp[folder+'_'+key] = js_tmp[key]
                del js_tmp[key]
            js_full.update(js_tmp)
        return js_full
    
    def stand_dict(self, json_file):
        data_set = {}
        for j_f in json_file.keys():
            region = j_f.split('/')[-2]
            if region not in data_set.keys():
                data_set[region] = {}
                data_set[region]['all_stands'] = []
            img_id = j_f.split('/')[-1][:-1-len(j_f.split('/')[-1].split('_')[-1])]
            if img_id not in data_set[region].keys():
                data_set[region][img_id] = {}
            stand_id = j_f.split('/')[-1].split('_')[-1]
            if stand_id not in data_set[region][img_id].keys():
                data_set[region][img_id][stand_id] = {}
                data_set[region][img_id][stand_id]['area'] = json_file[j_f]
                data_set[region][img_id][stand_id]['key'] = j_f
                data_set[region]['all_stands'].append(stand_id)
        return data_set
        
    def train_val_split(self, json_file, split_ration):
        data_set = self.stand_dict(json_file)
                
        seed(1)
        train_samples, val_samples = {}, {}
        for region in data_set.keys():
            for stand_id in set(data_set[region]['all_stands']):
                put_in_train = random.random() < split_ration
                for img_id in data_set[region].keys():
                    if img_id == 'all_stands':
                        continue
                    if stand_id in data_set[region][img_id].keys() and put_in_train:
                        train_samples[data_set[region][img_id][stand_id]['key']] = data_set[region][img_id][stand_id]['area']
                    elif stand_id in data_set[region][img_id].keys():
                        val_samples[data_set[region][img_id][stand_id]['key']] = data_set[region][img_id][stand_id]['area']   
        
        return train_samples, val_samples
    
    def load_dataset(self, folders, json_name_cl0, json_name_cl1, folders_val = None, split_ration=0.7):
        self.json_file_cl0_train = self.read_json(folders, json_name_cl0)
        self.json_file_cl1_train = self.read_json(folders, json_name_cl1)
        
        if self.cloud:
            self.json_file_cl0_train = self.drop_cloud(self.json_file_cl0_train)
            self.json_file_cl1_train = self.drop_cloud(self.json_file_cl1_train)
        
        if folders_val != None:
            self.json_file_cl0_val = self.read_json(folders_val, json_name_cl0)
            self.json_file_cl1_val = self.read_json(folders_val, json_name_cl1)
            if self.cloud:
                self.json_file_cl0_val = self.drop_cloud(self.json_file_cl0_val)
                self.json_file_cl1_val = self.drop_cloud(self.json_file_cl1_val)
        else:
            self.json_file_cl0_train, self.json_file_cl0_val = self.train_val_split(self.json_file_cl0_train, split_ration)
            self.json_file_cl1_train, self.json_file_cl1_val = self.train_val_split(self.json_file_cl1_train, split_ration)
        
    def drop_cloud(self, json_data):
        keys_list = set(json_data.keys())
        for key in keys_list:
            upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(json_data[key])
            # drop big crops
            if pol_width > 128 or pol_height > 128 or pol_width < 10 or pol_height < 10: # 10
                del json_data[key]
                continue
            with rasterio.open(key[:-len(key.split('_')[-1])-1] +'/B02.tif') as src:
                size_x = src.width
                size_y = src.height
            difference_x = max(0, self.IMG_COL - int(pol_width))
            difference_y = max(0, self.IMG_ROW - int(pol_height))
            rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),min(size_x, 
                                                         int(upper_left_x) + int(pol_width) + difference_x) -
                                  self.IMG_COL)
            rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),min(size_y, 
                                                         int(upper_left_y) + int(pol_height) + difference_y) -
                                  self.IMG_ROW)

            window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
        
            with rasterio.open(key[:-len(key.split('_')[-1])-1] + '/SCL.tif') as src:
                cloud_mask = src.read(window=window).astype(np.uint8)
            cloud_mask = (cloud_mask == 4).astype('uint8') 
            kernel = np.ones((15,15),np.uint8)
            cloud_mask = cv2.erode(cloud_mask,kernel,iterations = 1)
            if np.sum(np.where(cloud_mask==0,1,0)) > self.IMG_ROW*self.IMG_COL / 10:
                del json_data[key]
        return json_data
"""
Module for neural images dataset.
"""

import logging
import os
import tensorflow as tf
import scipy.io
import cv2 
import numpy as np 



logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG) 

BUCKET_URL = 'https://storage.googleapis.com/serrelab-public/neural_recordings/arcaro_2020.zip'

class Neural_dataset():
    """
    Class for neural images dataset.
    """
    def __init__(self):
       self.load_data()  

    def download_to_local(self):
        """
        Download the data from the bucket to local.
        """
        logging.info('File download Started…. Wait for the job to complete.')
        # Create this folder locally if not exists
        path_to_downloaded_file = tf.keras.utils.get_file(
            origin=BUCKET_URL,
            extract=True,
        )
        logging.info(f'File download completed. Check {path_to_downloaded_file} for the downloaded zip file.')
        self.path_to_downloaded_file = path_to_downloaded_file
        return path_to_downloaded_file
    
    def preprocess(self):
        """
        Preprocess the data. 
        """

        X = self.original_stimuli.copy()
        _mean_imagenet = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=tf.float32)
        _std_imagenet =  tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=tf.float32)
        X = np.array(X, np.float32) / 255.0
        X -= _mean_imagenet
        X /= _mean_imagenet
        return X
    
    def load_data(self):
        """
        Load the data. 
        
        The relevant info is in 'imageRF'
            dim 0 &1 : XY locations 
            dim 2 time in ms .. window 100-150-ms George, 150-300ms Red
            dim 3 neural channels
                george 1-32 are posterior IT and 33-63 are central IT
            dim 4 images
        """
        
        logging.info('Downloading data from bucket.')
        path_to_downloaded_file = self.download_to_local()
        logging.info('Data downloaded.')

        george_mat = scipy.io.loadmat(f'{path_to_downloaded_file}/George/george_060118_data.mat')
        red_mat = scipy.io.loadmat(f'{path_to_downloaded_file}/Red/red_062419_data.mat')
    
        george_data = george_mat['imageRF'].transpose((-1,0,1,2,3))# Transposing, so images are first. 
        red_data = red_mat['imageRF'].transpose((-1,0,1,2,3))
        
        self.george_posterior_it = george_data[:,:,:,:,:33]
        self.george_central_it = george_data[:,:,:,:,33:]
        self.red_central_it = red_data 

        logging.info('George posterior IT:',self.george_posterior_it.shape)
        logging.info('George central IT:',self.george_central_it.shape)
        logging.info('Red central IT:',self.red_central_it.shape)

        self.george_files = [f'{path_to_downloaded_file}/George/natural_test/'+george_mat['UniqueNames'][0][i][0].split('\\')[-1] for i in range(george_data.shape[0])]
        self.red_files =  [f'{path_to_downloaded_file}/Red/red_naturalface/'+red_mat['UniqueNames'][0][i][0].split('\\')[-1] for i in range(red_data.shape[0])]
        
        all_files = self.george_files + self.red_files 
        X = np.array([cv2.resize(cv2.imread(all_files[i])[:,:,::-1], (224, 224)) for i in range(len(all_files))], np.uint8)

        self.original_stimuli = X
        self.preprocessed_stimuli = self.preprocess()
        self.original_stimuli_red = self.original_stimuli[-len(self.red_files):]
        self.original_stimuli_george = self.original_stimuli[:len(self.george_files)]
        self.preprocessed_stimuli_red = self.preprocessed_stimuli[-len(self.red_files):]
        self.preprocessed_stimuli_george = self.preprocessed_stimuli[:len(self.george_files)]
        logging.info('Data loaded.')
    
    
    def get_george_posterior_it_temp(self,a=50,b=90):
        return self.george_posterior_it[:,::-1,::-1,a:b,:]

    def get_george_central_it_temp(self,a=50,b=90):
        return self.george_central_it[:,::-1,::-1,a:b,:]
    
    def get_red_central_it_temp(self,a=50,b=90):
        return self.red_central_it[:,::-1,::-1,a:b,:]
    
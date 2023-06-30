from neural_harmonizer.data.dataset import Neural_dataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def test_dataset():
    """
    Test the dataset class.
    """
    dataset = Neural_dataset()
    dataset.download_to_local()
    dataset.load_data()
    dataset.preprocess()
    dataset.get_george_posterior_it_temp()
    dataset.get_george_central_it_temp()
    dataset.get_red_central_it_temp()
    return dataset
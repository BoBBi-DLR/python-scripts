#!/bin/python3

#########################################################################################
# Date: 2023-06-10
# Creator: Pascal Müller
# Version:0.1
#########################################################################################

import tensorflow as tf

#-----------------------------------------------------------------------------------------
# GPU_Runner
#-----------------------------------------------------------------------------------------

class GPU_Runner:
    
    #---------------CONSTRUCTOR---------------
    def __init__(self):
        # Initialize TensorFlow and check for GPU availability
        self.physical_devices = tf.config.list_physical_devices('GPU')
        self.is_gpu_available = len(self.physical_devices) > 0
    #-----------------------------------------

    def run_on_gpu(self, code):
        if not self.is_gpu_available:
            raise Exception("No GPU available.")

        # Run the provided code on the GPU
        with tf.device('/GPU:0'):
            result = code()

        return result
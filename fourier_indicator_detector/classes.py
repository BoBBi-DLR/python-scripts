#!/bin/python3

#########################################################################################
# Date: 2023-06-10
# Creator: Pascal Müller
# Version:0.1
#########################################################################################

import math
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
#from drawnow import drawnow, figure
#from pylab import *
#from IPython.display import clear_output
import cv2
import time
#from GPU_Runner import GPU_Runner
from CPUMultiTasker import CPUMultiTasker
import concurrent.futures

#-----------------------------------------------------------------------------------------
# Fourier_Indicator_Detector
#-----------------------------------------------------------------------------------------

class Fourier_Indicator_Detector:

    
    #---------------CONSTRUCTOR---------------
    def __init__(self, direction, color_channel=2):

        if self.check_inputs(direction, color_channel):
            self.direction = direction
            self.color_channel = color_channel
    #-----------------------------------------     

    def check_inputs(self, direction, color_channel):
        
        if direction in ['right','left'] and color_channel in [0,1,2,3]:
            return True
        else:
            print(ValueError(direction, color_channel))
            return False

    #-----------------------
    # tested
    def extract_color_channel(self, frame_in):

        # extract only the desired color cannel(s) 
        if self.color_channel in range(0,3):
           
           frame_out = frame_in[:,:,self.color_channel]

        # to be implemented for red+green channel
        else:
            print('no valid color channel')
            pass

        return frame_out

    #-----------------------

    def calculate_fft(self, video, i, mode='sum'):
        
        frames_1channel=[]
        for frame in video.cropped_frames[:i+1]:
            frame_1channel = self.extract_color_channel(frame)
            frames_1channel.append(frame_1channel)

        # only mode implemented yet
        if mode == 'sum':
            summed_frames=[]
            for frame in frames_1channel:
                summed_frames.append(np.sum(frame))
            
            """ # run FFt on GPU
            gpu_runner = GPU_Runner()
            fft_amps = gpu_runner.run_on_gpu(fft(summed_frames))
            fft_freqs = gpu_runner.run_on_gpu(fftfreq(len(fft_amps), 1/video.samplerate)) """

            # run on CPU
            fft_amps = fft(summed_frames)
            fft_freqs = fftfreq(len(fft_amps), 1/video.samplerate)

        # may be implemented: single pixel or cluster
        return fft_freqs, fft_amps, frames_1channel[-1]

   #-----------------------

    def plot_single_frame(self, frame_1channel, fft_amps, fft_freqs, ax1, ax2):

            #ax1.subplot(1,2,1)
            ax1.imshow(frame_1channel, cmap='gray')

            #ax2=fig.add_subplot(1,2,2)
            ax2.clear()  # Clear the previous FFT plot
            ax2.set_xlim(0, 6.1)
            ax2.set_ylim(0, 0.1*np.max(fft_amps))
            ax2.plot(fft_freqs, np.abs(fft_amps))

            # Add vertical lines for first harmonic wave
            ax2.axvline(x=1.0, linestyle=':', color='red', label='1 Hz')
            ax2.axvline(x=1.5, linestyle='--', color='red', label='1.5 Hz')
            ax2.axvline(x=2.0, linestyle=':', color='red', label='2 Hz')

            # Add vertical lines for second harmonic wave
            ax2.axvline(x=3.0, linestyle=':', color='blue', label='3 Hz')
            ax2.axvline(x=4.5, linestyle='--', color='blue', label='4.5 Hz')
            ax2.axvline(x=6.0, linestyle=':', color='blue', label='6 Hz')

            plt.pause(0.001)  # Pause briefly to update the plot

    #-----------------------

    def plot_furier_analysis(self, video):

        tasker = CPUMultiTasker()

        if not isinstance(video, Video):
            print(f'the argument video has to istance of the Video class!')

        else:
            plt.ion()
            fig = plt.figure()
            
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('Amplitude (tbd)')
            
            #ax2.set_ylim(0, 1)

            for i in range(1000, len(video.cropped_frames)+1):

                #print(f'calculating FFT for frame {i}')


                # run on only 1 CPU Core
                # do FFT for all frames till i/fps of the video
                fft_freqs, fft_amps, frame_1channel = self.calculate_fft(video, i)

                #ax2.set_ylim(0, 1.2*max(fft_amps))

                #print(f'plotting frame {i} and FFT')
                self.plot_single_frame(frame_1channel, fft_amps, fft_freqs, ax1, ax2)
                
                """ # run on all CPU Cores (method 1)
                fft_freqs, fft_amps, frame_1channel = tasker.run_function(self.calculate_fft, args=(video, i))
                tasker.run_function(self.plot_single_frame, args=(frame_1channel, fft_amps, fft_freqs, ax1, ax2)) """

                """ # run on all CPU Cores (method 2)
                with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                    future = executor.submit(self.calculate_fft, video, i)
                    results = future.result()
                    fft_freqs =results[0]
                    fft_amps = results[1]
                    frame_1channel = results[2]
                    executor.submit(self.plot_single_frame, frame_1channel, fft_amps, fft_freqs, ax1, ax2)"""

                # time.sleep(1/video.fps)     # to be optimized. Check if FFT is fast enough so this approximation is valid and the video plays in realtime
            
            video.fft_freqs = fft_freqs
            video.fft_amps = fft_amps

            plt.ioff()

            plt.show()
    #-----------------------


#-----------------------------------------------------------------------------------------
# Video
#-----------------------------------------------------------------------------------------

class Video:

    #---------------CONSTRUCTOR---------------
    def __init__(self, filename, samples_per_period=100, fft_amp=[], fft_freq=[]):
        self.filename = filename
        self.__samples_per_period = samples_per_period
        self.samplerate = int
        self.frames = []
        self.cropped_frames = []
        self.fps = []
        self.fft_amps = fft_amp
        self.fft_freqs = fft_freq
        self.indicator_detected = False

        self.__read_frames()
        self.__import_bounding_boxes()

        self.cropped_frames =self.frames    #temporary for testing!!!!
    #-----------------------------------------
       
    def __read_frames(self):

        video= cv2.VideoCapture(self.filename)
        self.fps = video.get(cv2.CAP_PROP_FPS)

        # Check if the video file was successfully opened
        if not video.isOpened():
            print("Error opening video file")

        while video.isOpened():

            # Read the next frame from the video
            ret, frame = video.read()

            # Check if a frame was successfully read
            if ret:

                # convert frame to numpy array
                frame = np.array(frame)
                
                # append frame to frames array and add as many of the current frame
                # that a minimum 100 frames per period are achieved 
                # based on indicator flashing with 1.5 Hz +- 0.5 Hz) 
                if self.fps < 200:
                    count = math.ceil(self.__samples_per_period*2/self.fps)
                else:
                    count = self.indicator_detected = False

                self.samplerate = self.fps*count
                for i in range(0, count):
                    self.frames.append(frame)
                  
            else:
                break
                
        # Release the video object and close any open windows
        video.release()

    #-----------------------

    def play_video(self):
        """ plt.figure()
        for frame in self.cropped_frames:
            plt.imshow(frame)
            time.sleep(1/self.samplerate) """
        
        for frame in self.cropped_frames:
            cv2.imshow('Frame', frame)
            time.sleep(1/self.samplerate)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        

    #-----------------------

    def crop_frames(self):

        # to be implemented
        if not np.len(self.frames) == np.len(self.bounding_boxes):
            print(f'Error: array with frames and array with bounding boxes are not the same length')
        else:
            return

    #-----------------------

    def __import_bounding_boxes(self):
        # to be implemented
        return
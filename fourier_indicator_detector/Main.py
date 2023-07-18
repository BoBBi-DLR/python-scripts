#!/bin/python3

#########################################################################################
# Date: 2023-06-10
# Creator: Pascal Müller
# Version:0.1
#########################################################################################

from classes import Video, Fourier_Indicator_Detector

def main():
    print('loading mp4-file')
    vid1 = Video('test2.mp4', 100)
    print('mp4-file loaded')  

    print('initialize detector')
    detector = Fourier_Indicator_Detector('right')
    print('detector initialized')    

    detector.plot_furier_analysis(vid1)

if __name__ == "__main__":
    main()
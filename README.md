# Drivable-Road-Detection
This code segments out the drivable portion of the road from the surrounding with the help of inbuilt open-CV library.

Installations required:

1.Numpy library

2.OpenCV library(cv2)

3.Matplotlib library

4.Python 3.6

Now road-lane detection techniques exists but they do not work for Indian-roads due to the fading of lanes and unsmooth road surfaces.
So, this project directly takes use of the pixel-values of the road and applies different opencv functions to segment out the drivable portion of the road.

Now, we can deal with pixel values in 2 different ways- OpenCV library for image processing or using Neural Networks.

Output images of particular frames of 2 videos are here below:

Frame 1:

![1](https://github.com/shahjui2000/Drive-able-Road-Detection/blob/master/frame1.PNG)



Frame 2:

![2](https://github.com/shahjui2000/Drive-able-Road-Detection/blob/master/frame2.PNG)



Since, the output shows great acuracy , using openCV for segmentation becomes a better way since, Neural Networks come with a lot of computations and costs.


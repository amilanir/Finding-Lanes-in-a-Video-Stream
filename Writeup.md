
# Finding Lane Lines on the Road

The goals/steps of this project are the following:

- Make a pipeline that finds lane lines on the road.

# Reflections

## Description
In this pipeline which identifies lanes lines uses the following functions to actively identify lanes lines inan image and plot lane liine in the image. 

- **grayscale**: Returns a gray scaled version of the input image using **cv2.cvtColor** method.
- **canny**: Use a [Canny transformation](https://en.wikipedia.org/wiki/Canny_edge_detector) to find edges on the image using **cv2.Canny** method.
- **gaussian_blur**: Applies a Gaussian blur to the provided image using **cv2.GaussianBlur** method.
- **region_of_interest**: Eliminate parts of the image that are not interesting in regards to the line detection
- **draw_lines**: Draws `lines` with `color` and `thickness`.Lines are drawn on the image inplace
- **hough_lines**: Use a [Hough transformation](https://en.wikipedia.org/wiki/Hough_transform) to find the lines on the masked image using **cv2.cv2.HoughLinesP**. It also adjust a line to the set of lines returned by the Hough transformation in order to have a clearer-two-lines representation of the road lines using **np.polyfit** method.
- **weighted_img**: Merges the output of **houghAction** with the original image to represent the lines on it.
- **construct_lane_lines**: This function filters out lines and assign lines to the left or right lane group according to their slope. The line equation is: y = m*x + b.
When we detect a line with a positive slope and a value greater than the positive threshold it gets assigned to the left group, if the slope is negative and smaller than the negative threshold it gets assigned to the right group.
At the end we do a linear regression line fitting for both the left and right lane lines.
    

## Potential shortcomings/suggestions

- The lines shake a lot on the videos, a better way to average them should be possible.
- The line size should be improved.
- Bright points outside the lines were taking into account.

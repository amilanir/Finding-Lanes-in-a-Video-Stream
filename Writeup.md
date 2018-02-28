
Creating a Great Writeup
---
For this project, a great writeup should provide a detailed response to the "Reflection" section of the [project rubric](https://review.udacity.com/#!/rubrics/322/view). There are three parts to the reflection:

1. Describe the pipeline

2. Identify any shortcomings

3. Suggest possible improvements

We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md). 


The Project
---

## If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go!   If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  

**Step 3:** Complete the project and submit both the Ipython notebook and the project writeup

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).



# Finding Lane Lines on the Road

The goals/steps of this project are the following:

- Make a pipeline that finds lane lines on the road.
- Reflect on your work in a written report.

# Reflections

## Description
The pipeline consists on six steps represented by six different functions:

- **grayAction**: Returns a gray scaled version of the input image using **cv2.cvtColor** method.
- **blurAction**: Applies a Gaussian blur to the provided image using **cv2.GaussianBlur** method.
- **cannyAction**: Use a [Canny transformation](https://en.wikipedia.org/wiki/Canny_edge_detector) to find edges on the image using **cv2.Canny** method.
- **maskAction**: Eliminate parts of the image that are not interesting in regards to the line detection (for now...).
- **houghAction**: Use a [Hough transformation](https://en.wikipedia.org/wiki/Hough_transform) to find the lines on the masked image using **cv2.cv2.HoughLinesP**. It also adjust a line to the set of lines returned by the Hough transformation in order to have a clearer-two-lines representation of the road lines using **np.polyfit** method.
- **weighted_img**: Merges the output of **houghAction** with the original image to represent the lines on it.

To average and interpolate the lines on the **construct_lane_lines** function, I try to find the lines going on the left and right side of the image. Each point of those sides were accumulated and first degree polynomial was fit to those points using **numpy.polyfit**. When the line equation was found, it was evaluated on the region of interest to define the points where the line would be shown on the image.

## Potential shortcomings/suggestions

- The lines shake a lot on the videos, a better way to average them should be possible.
- The line size should be improved.
- Bright points outside the lines were taking into account.

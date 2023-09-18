# 3d Mouse Tracking

This repository contains code designed to track a set of predefined mouse body parts in 3d from four simultaneous videos taken of the mouse as it moves around an arena. This project was done as a Summer project at the request of Dr. Gregory Schwartz, PhD and Devon Greer, a PhD candidate of his, through the [Schwartz Lab](http://schwartzlab.feinberg.northwestern.edu/), which is in the [Department of Ophthalmology](https://www.feinberg.northwestern.edu/sites/ophthalmology/) at Northwestern's Feinberg School of Medicine. It is additionally associated with the [Northwestern University Interdepartmental Neuroscience](https://www.nuin.northwestern.edu/) (NUIN) program. 

## What's In Here?

The most significant contributions of this repository can be found in the following files:
* `./model.py`
* `./projector.py`
* `./calibrate_extrinsics.py`

### model.py

This file contains a PyTorch `nn.Module` class called `Model`.

It is primarily a three dimensional Convolutional Neural Network that accepts four channels as there are four input camera views and outputs eight channels as there are eight body parts that we wish to track. It consists of eight [convolutional layers](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html) and eight [pooling layers](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html?highlight=maxpool3d#torch.nn.MaxPool3d), with a pooling layer following each convolutional layer. The specific parameters that each layer employs can be found in the file. Note that the default parameters specified therein are a product of the known input dimensions (we'll discuss the input at the end of [the following section](#why-this-model?)), and these parameters may need to change if the input dimensions change drastically.

Following the convolutional and pooling layers we have two linear layers that exist to process the semantic information provided by the convolutional network into 3d points for each body part. 

These outputs are then fed into the model's loss function, which you can find in this file as well. This function is primarily designed to compute the MSE of the Euclidian distance between the projection of these discovered 3d points into each camera's image coordinate system (pixels) and ground truth values.

#### Why this model?

At the beginning of this project, the plan was to leverage [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), a open-source library for tracking body parts in 2d across a video. There exist extensions of DeepLabCut that can triangulate discovered 2d points across multiple camera views, like [Anipose](https://anipose.readthedocs.io/en/latest/) but it requires that every key-point be viewed by two cameras at all times. Due to the setup of our arena, this was very much not the case. 

Furthermore, the side cameras have notably poor resolution, along with significant distortion.

Here is an example of an image from one of the side cameras showcasing a particularly low resolution image followed by one showcasing the high distortion.

![Low Resolution Image](/readme_images/Low-Resolution-Sample.png)

![High Distortion Image](/readme_images/High-Distortion-Sample.png)

This indirectly resulted in inconsistent tracking with DeepLabCut as it led to inconsistent labeling when it came to creating the training data set for DeepLabCut. Often times, there were other mice behind the clear tall panes you can see above and it is sometimes hard for a human labeler to tell where the mouse in question ends and the mouse behind the pane begins. This difficulty was magnified by the lighting of the arena (which was only from the top) and the pigment of the mice used (very dark). These complications made it hard to accurately label certain body parts of the mouse.

![Top Image](/readme_images/Top-Sample.png)

The top camera was much higher resolution, and due to its relative location, it did not suffer from as many overlaps between the mice in the arena and the mice behind the panes.

Of course, given perfect camera parameters and prior measurements of the average distance between the body parts in question, one could remove significantly unlikely measurements (from DeepLabCut) and triangulate using least-squares on the remaining data. Given the low key-point coverage in this context, this would produce a minimal but potentially adequate representation of the 3d pose of the mouse across time. However, the camera parameters I was given were not perfect:

![Projections](/readme_images/projections.png)

These pictures show the result of projecting 5 points on the floor of the arena, one in the very middle and one offset in each direction by the radius of the inner tube along with one point in the very middle of the arena but pushed straight up towards the camera by about 15 inches. The projection was done using OpenCV's `projectPoints()` implementation. As you can see, from the top camera, these points are more or less correct. This is because the top camera's coordinate system was taken to be the world coordinate system, so the accuracy of its projections was solely dependent on its intrinsic parameters, which appear to be mostly correct. Note that the top camera has minimal distortion. 

However, as we can see in the other pictures, the projections are consistently offset towards the camera. This represents a degree of error that would make triangulation using least-squares infeasible as solutions would not be representative of true 3d points. 

From here, there are two ways to approach the problem. 

1. We can try to fix the camera parameters.
2. We can try to triangulate anyway and then offset discovered 3d points by some derived mapping between true 3d points and skewed 3d points.

The second idea here is interesting, but as far as I was able to determine, impossible, or at least impractical. To do so, we would first need to get some understanding of true 3d points. We can accomplish this by labeling consistent points on the mouse from each camera view. Then, we can un-project these points into rays in 3d world coordinates, triangulate using least-squares or something similar, and re-project them back into image coordinates. This gives us a mapping from ground truth 2d image points to skewed 2d image points. Unfortunately, this mapping is not invertible as there can be many possible sets of rays that will converge to the same triangulated 3d point (the function is not one-to-one). If it were, we could apply its inverse to each ground truth 2d label and teach a model to learn using these skewed points as truth. A method of creating a one-to-one and onto function (hence, invertible) that can accurately represent this distortion would conceivably solve this problem and is an interesting avenue for further research.

So, I set out to fix the camera parameters. Specifically, the extrinsic camera parameters. For an in depth description of the applied methodology and complications, see: [calibrate_extrinsics.py](#calibrate_extrinsics.py).

Unfortunately, despite extensive parameter sweeping, I was unable to determine a good enough set of extrinsic camera parameters. Accordingly, I decided to use a model that I believe to be most likely to work if it were to be given better extrinsic parameters. While a least-squares implementation may work, it would necessarily need to minimize the difference between expected joint length and observed joint length given the inconsistency of DeepLabCut's labels for the side camera, and the interplay between minimizing projection error and joint error in this context could produce biased results. More importantly, it could produce significant error as the joints are not necessarily constant length, nor are their measurements necessarily accurate. 

So, I decided to use a model that could utilize the 3d structure of the mouse to make inferences without relying on exact 3d joint measurements. This model was inspired by a [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_BKinD-3D_Self-Supervised_3D_Keypoint_Discovery_From_Multi-View_Videos_CVPR_2023_paper.pdf) shared with me by a previous professor of mine, Emma Alexander, PhD. 

This paper uses a self-supervised model, which is not required in this context. So, I simplified it drastically by introducing a manually labeled dataset. The dataset consists of 267 sets of 4 images with 8 labeled body parts per image.

The general idea behind this model is to construct a voxel space consisting of four channels and assign values to voxels in channel `i` in accordance with the bilinear interpolation of the projection onto camera `i`. This voxel space is then what is passed into the 3d convolutional network which infers 3d points and projects them onto ground truth values. Given enough labeled data, which I believe I have provided, and good enough extrinsic camera parameters, I am confident this model will work. 

This model may also accept known edge distances if so desired. This functionality can be integrated by adding the correct information to [body_parts.json](#body_parts.json). The information that currently exists therein is dummy information.

### projector.py

This file is notable even when considered independently of this project. It consists primarily of a `torch.autograd.Function` class called `ProjectFunction`. Much of the time spent on this project consisted of determining how to correctly incorporate 3d->2d point projection into PyTorch. Ultimately, after much trial and error, I decided to piggy back off the functionality provided by OpenCV's `ProjectPoints()` function. To do this, I wrapped their function within the ProjectFunction class's forward pass, and implemented its associated [vector-Jacobian product](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) in the backward pass. 

This allows any model I implement in PyTorch that projects to compute the necessary partial derivatives through this projection. Wonderfully, OpenCV provides jacobian information with respect to the inputted camera parameters. Not wonderfully, it does not provide similar information for the input parameters. So, instead of analytically calculating these derivatives, I estimated them numerically. This seemed very reasonable to me as minute changes in an inputted 3d point should not result in drastic changes in the resulting projected 2d point. 

Note that for the purposes of this project the only projection parameters I wanted to update according to some loss were the rotation and translation vectors, but the provided code could be relatively easily extended to propagate vector-Jacobian products for the other camera parameters as well if so desired.

### calibrate_extrinsics.py

To calibrate the extrinsic camera parameters, 50 consistently labeled and double checked set of four labels were used as ground truth values to compare against their reprojection. Specifically, I used the non-linear Levenberg-Marquardt optimization algorithm, which is often used for this purpose, to implement bundle adjustment for calibration. I used [Torchimize's least-squares Levenberg-Marquardt](https://github.com/hahnec/torchimize/blob/master/torchimize/functions/single/lma_fun_single.py) implementation and provided it a function that takes the ground truth labels, un-projects them into rays in 3d world coordinates, triangulates a 3d position using least-squares, and projects them back into image coordinates. Both the un-projection and the projection rely on the extrinsic camera parameters, and using my [projector](#projector.py) implementation, the jacobian is carried throughout these calculation to support optimization. Unfortunately, there were some complications

#### Calibration Complications

Starting from the given extrinsics, this process terminated quickly and simply did not work. This is because it got stuck in a local optimum, a common fallback of Levenberg-Marquardt. 

To attempt to ameliorate this, I ran parameter sweep on the "tau" parameter and the initial extrinsic parameters. Despite extensive search, this process proved unsuccessful given the time frame of this project. It is possible that further sweeping would produce better parameters.

There are a few possible reasons for this:

1. The training data used for this process had bias in the labels or too high variance with too little data.
2. The intrinsic camera parameters are incorrect.
3. The implementation of Least-Squares Levenberg-Marquardt from [Torchimize](https://github.com/hahnec/torchimizehttps://github.com/hahnec/torchimize) I used was incorrect.
4. My implementation of computing re-projection error is incorrect.
5. OpenCV's projectPoints() implementation is incorrect.

In order to avoid labeling bias as much as possible, I used about 50 consistently labeled and double checked sets of four labels, (one for each camera), of the mouse's tail, as it is the most precise point on the mouse. I also used this amount of data because the computational complexity of parameter sweeping proved to be restrictive, and I thought that to be the more important aspect at the time. I have since labeled 267 sets of 8 sets (one for each body part) of four labels. On initial attempts this data did not help, but further attempts may be successful.

It is also very possible the intrinsic camera parameters are incorrect. After testing the parameters for the intrinsic camera matrix, it seems like these are correct. However, attempts to un-distort images from the side cameras resulted in imperfect results:

![Undistorted Image](/readme_images/Undistortion-Test.png)

Better intrinsic parameters, particularly distortion coefficients, may prove to helpful as well. 

## Getting Started

This project is written in Python 3.9.10. However, any version of Python >= 3.6 should be just fine. To install this project on your local machine, it is recommended that you create a virtual environment by running the following command in your terminal:

	python3 -m venv ./venv_name

After this, make sure to activate the environment like so:

	source ./venv_name/bin/activate

This will run the "activate" binary file, and you should see 

	(venv_name) youruser@yourcomputer cur_directory % 

Or something very similar. All of the above is based on downloading to a mac, but it shouldn't be very different on other platforms. 

Once you've done this you can easily install the necessary packages to your virtual environment like so:

	pip3 install -r ./requirements.txt

Finally, you must install ffmpeg on your machine to use some of the code that interacts directly with video files. If you have homebrew installed you can do this easily:

	brew install ffmpeg

At this point, after you've unzipped all the .zip files or set up your own data, you should be able to run any of the python files in this repository like so:

	python3 -m python_file_name

Note that you should not include the ".py" file extension when running this command.

Even after compression the .zip files containing the video and training data for this project exceed GitHub's file size limits. The rest of this readme assumes that these files have been downloaded and un-zipped in your working directory (./All_Data.zip and ./Training_Data.zip).

Of course, you can set up your own data, which will also be explained [below](#data-wrangling).

But, if you'd like access to the data associated with this project, feel free to reach out to me at patrickdwy@icloud.com, and I will happily send the files to you.

> ### An Aside
> 
> This project contains a directory `./notebooks` which contain most of the same logic and functionality you'll find throughout this project, but in a much less readable format. I did some of my research and proof of concepts in these notebooks, and then I translated these notebooks to Python files that are more modular. This folder simply represents the final iteration of my exploratory work, and there is much more code I've written for this project that isn't useful at this point. As an exception, the file `./notebooks/Camera_Calibration.ipynb` is a notebook I wrote specifically to run on Google Collab to leverage their gpu interface. 
> 
> Note that all of the PyTorch code in this project is designed to notice whether or not your machine has a Cuda gpu available and act in accordance. For reference, this works with the following simple line:
> 
> `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`

## Setup

In order to use the code provided herein, you must have a video dataset.

This dataset must follow the following structure:

```
/All_Data
	/trial1
		/cam1.videofile
		/cam2.videofile
		/cam3.videofile
		/…
	/trial2
		/cam1.videofile
		/cam2.videofile
		/cam3.videofile
		/…
	/…
```

Where `videofile` is any video extension supported by ffmpeg. Note that the naming of these video files does not matter except that the order of the list that contains there names and is sorted by Python's .sort() matches up with the order of the list that contains the names of cameras (specified in [camera_parameters.json](#camera_parameters.json)) after it too is sorted the same way. To accomplish this easily, name each file `{name-of-camera}.{video-file-extension}`.

The only other requirement is that each set of videos starts simultaneously. 

`./All_Data.zip` contains an example of these requirements in the original context of this project.

### Data Wrangling

Once you've got your video files set up, you can either use the training data provided in `./Training_Data.zip`, or you can create your own. To do this, you can use the conveniently named `./create_training_data.py`.

This file has default arguments that you can override easily by passing command line key-value arguments. 

Specifically, this file needs to know the directory where the video files are located, the directory to output the training data, the number of images to create, and the method to select images. The default values for the parameters are `./All_Data`, `./Training_Data`, the integer 10, and the string "even", respectively. 

To change num_images to n, add `num_images=n` as a command line argument. 

To change the directory specifying the video locations to a directory within `./` with name "my-dir", add `in-dirname=my-dir` as a command line argument. If "my-dir" is an absolute path, add `in_dir_absolute=my-dir` instead.

To change the directory specifying the output location of the training data, follow the directions in the paragraph directly above this one while replacing "in" with "out".

Although this is not recommended, you may also change this file to produce frames selected randomly from a uniform distribution. To do this, pass `method=random` as a command line argument.

### Data Labeling

You can find labels for training data provided in this repository in `./Training_Data`. Each directory within `./Training_Data` contains a set of frames taken at the same time, without respect to the set of videos they were taken from. The labels can be found in these directories as `./Training_Data/{trial-dir}/labels.csv`.

If you have your own training data or would like to label your training data for any reason, you can do that using the convenient `label_training_data.py`.

This file keeps track of a global state in `./labeling_state.csv`, which it will create if it does not exist (so you can restart labeling by deleting the file, or by manually changing its value). 

This allows you to label images intermittently. It will label trials in the training data directory in alphabetical order according to Python's `.sort()` function. 

For each trial, it will prompt you with images in the same order. You can then label the pixel location of body parts in the order prescribed in [body_parts.json](#body_parts.json) by left clicking on the body part in the image. If you do see the body part in the image, right click. This will store `(-1,-1)` instead of the pixel location of your mouse as these values are used to filter data later. Each time you click, you will see in the console describing what you did. If you make a mistake, simply press delete as many times as you need, it will show you the current state of labels for that image. If you do not see any of the body parts in the image you're shown, press space. Finally, press enter to move to the next image. The program will only do so when you press enter. Unfortunately, you can't easily go back once you press enter, so please make sure your labels are correct before you do so. Of course, you can always go back to the beginning of the trial you're on by quitting the program. You'll know the list is full when you click and the console says "List Full!". It will also tell you which body part is next whenever you click or delete. If you'd like to see the full list at any point, just press the 's' key.

### Camera Extrinsic Calibration & Testing

If you've got labeled training data, all you need is calibrated cameras. 

You can find the parameters that were initially provided for this project in [camera_parameters.json](#camera_parameters.json) under `"current_params"`. 

While the camera calibration routine is running, a counter will keep track of the number of different values it has tried for the tau parameter, and this counter will be stored in the aforementioned json file. By default, each of these iterations will run 100 iterations of rvecs and tvecs that are randomly deviated from the given rvecs and tvecs. Each tau iteration, the program will record the set of camera parameters that scores the best according to the training set, and it will save then in the aforementioned json file under `"best_params"`. If you'd like to restart the calibration process, set counter to zero in this file.

## Other JSON Files

Information that is shared between files is stored in JSON files. 

### arena_info.json

This file contain 6 variables:

1. `"Max_Radius"`: Sets the size of the voxel space.
2. `"Points_Per_Inch"`: Sets the number of points per inch in the voxel space.
3. `"Arena_Height"`: Height of arena.
4. `"Inner_Radius"`: Radius of inner cylinder of arena.
5. `"Outer_Radius"`: Radius of arena.
6. `"Arena_Center_xy_from_Top"`: The x and y coordinates of the middle of the arena from the perspective of the top camera.

The center xy coordinates are not perfect, but are very close. I've added the notebook `./notebooks/Center_Finding_And_More` if you'd like to see how this was done.

### body_parts.json

This files contains 2 variables:

1. `"names"`: The list of body part names.
2. `"edges"`: List of edge information.

The first two values in each list in edges are the indices of the body parts in `"names"`. The third value is the length of this edge in 3d space, and the fourth is a relative weight to apply to this edge during learning.

## Author

* **Patrick Dwyer**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thank you Professor Emma Alexander, PhD, for connecting me with Greg Schwartz, PhD, and for providing valuable insights during this project.
* Thank you Ben Eckart, PhD, for discussing this project with me and providing valuable insights.
* Thank you Greg Schwartz, PhD, for giving me this opportunity to showcase my skills and knowledge.
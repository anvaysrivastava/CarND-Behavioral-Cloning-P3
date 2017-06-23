# CarND-Behavioral-Cloning-P3
Udacity CarND Behavioral Cloning Project https://www.udacity.com/drive

Template Repo: https://github.com/udacity/CarND-Behavioral-Cloning-P3


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md writeup for the project.
* video.mp4 https://www.youtube.com/watch?v=G0f-7z9AjwI.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 1x1, 3x3 and 5x5 filter sizes and depths between 1 and 16 (model.py lines 63-70)

The model includes ELU layers to introduce nonlinearity (model.py lines 63-70).

I have used MaxPooling between convolutional layers to reduce the training time without impacting performance.

I hae used a 1 depth 1x1 (line 63) filter so just after a normalized image so that road's colour should be given special significance, as was necessary for travelling on tarmac.


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17-22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I ran the network till the epoch of 10 and noticed when the val_loss rose up suddenly, showcasing an ideal epoch of 2. I could have used a Dropout layer, but was not able to notice any differnce on 2 epochs and chose to keep it out.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I the following dataset.
data : This was the dataset provided by udacity itself. Just this data was not enough as the car was sometimes not able to recover from a steep turn and also constantly going into the dirt tract that the fork.
manual_data : In order to teach car how to recover from steep turns, I did a lap around the track, constantly zig zagging to teach the car how to recover from corners.
smooth_driving : So that zig zag driving does not become the behavior of car. I did two laps trying to drive the car as center as possible.
dirt_curve1/2 : So that the car can give priority to use tarmac rather than dirt on the fork. I drove the car on that section two times.

I also made sure that the images are flipped too, since the tract is mostly turning towards left.
![alt text](https://github.com/anvaysrivastava/CarND-Behavioral-Cloning-P3/raw/master/resources/flipped.jpg)
![alt text](https://github.com/anvaysrivastava/CarND-Behavioral-Cloning-P3/blob/master/resources/normal.jpg)


Also there is a lot of environment variables that should not be fed into the network. For that I crop the image. 70 pixels from top and 25 from bottom.

![alt text](https://github.com/anvaysrivastava/CarND-Behavioral-Cloning-P3/raw/master/resources/flipped.jpg)
![alt text](https://github.com/anvaysrivastava/CarND-Behavioral-Cloning-P3/raw/master/resources/cropped.png)


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to see how the car performs on lenet.

Then I observed that car was able to drive nicely on straights, however at turns it would all of a sudden take steep turns and go out of the track. While validating the error I could see that validation error and training error w were almost same(~ 0.1). Hence I had to change the model. My first guess was to make the network a bit more wide.

While tweaking the models I realized that sometimes the model would plateau at error of (~0.4), higher than ~0.1 of lenet. Hence I thought the wide network should have more redundancy and I added a dropout.

Now the car was almost going fine apart from the forks of dirt track and if I forcefully make the car start from corner of road it would not recover very effectively.

Hence I added more data to the training set.

Then I normalized my input and chose to add more data set to reduce the number of epochs. After this the need for dropout also went away and the model became a lot more simple.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-93) consisted of a following

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | Trimmed image                                 |
| Convolution 1x1       | 1x1 stride, valid padding, 1 depth            |
| Convolution 3x3       | 3x3 stride, valid padding, 3 depth            |
| Max Pooling           | 2x2                                           |
| Convolution 3x3       | 3x3 stride, valid padding, 3 depth            |
| Max Pooling           | 2x2                                           |
| Convolution 5x5       | 5x5 stride, valid padding, 6 depth            |
| Max Pooling           | 2x2                                           |
| Convolution 5x5       | 5x5 stride, valid padding, 16 depth           |
| Flatten               |                                               |
| Fully connected       | 250                                           |
| Fully connected       | 100                                           |
| Fully connected       | 25                                            |
| Fully connected       | 1                                             |

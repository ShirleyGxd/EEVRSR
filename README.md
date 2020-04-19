# EEVRSR: End-To-End Convolutional Network for Video Rain Streaks Removal (ICIP 2019)

This is the source code of the [EEVRSR paper](https://ieeexplore.ieee.org/document/8803375):

## Abstract
Existing video rain streaks removal methods utilize various manual models to represent the appearance of rain streaks, and only use convolutional neural network (CNN) as a post-processing part to compensate the artifacts like misalignment caused by traditional de-raining operations. However, these manual models only work for some particular scenes because the distribution of rain streaks is complex and random. Moreover, since CNN network and previous traditional de-raining operations cannot be trained jointly, the output of CNN network may still contain artifacts. To address these problems, we propose an end-to-end video rain streaks removal CNN network called EEVRSR net. Experimental results of both synthetic and real data demonstrate that the proposed EEVRSR net achieves better performance in both speed and effectiveness over state-of-the-art methods.


## Prerequisites
1. python 3.6
2. cuda8.0 + cudnn5.0 + tensorflow 1.2.1

## Testing
1. Clone this repo.
2. Open the **test_code** and run **test_main.py**.
3. The derained result will be stored in the /test_code/EEVRSR_result/.

**Optional:**
You can choose the testing data by set the value of the variable **video_index**.
You can choose to use the whole EEVRSR net or only the de-raining sub-net by setting the value of the variable **net_derain_only_flag**.

## Model
The trained models are stored in final_model.
1. The EEVRSR_all stores the trained model of the whole EEVRSR net. 
2. The EEVRSR_derain_only stores the model of only the derain sub-net. 

## Data
Some testing data is stored in the test_data.





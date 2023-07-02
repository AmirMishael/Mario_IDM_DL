
### Labeling New Video



`dataset_extraction.py` use to label your own video with these 3 functions:

`extract_frames(video_path, frames_dir,start_sec,stop_sec)`: takes your video and decomposes it to frames. 

`create_metadata(frames_dir,metadata_path,model,group_size)`: labels the frames based on your model. saves a csv file with the frames titles and labels.
csv head: Up, Left, Right, B, ID, Image_path

## OpenAI Gym - Agent
We put our model into the big players' gym, OpenAI gym, so we can see what gameplay it produces.

`main_train_agent.py` use to train the agent based on our network in the OpenAI gym.

`main_gym_agent_play.py` lets the agent play the game.



##############################

# Break

# Super Mario Play
In this project we take the abilities of Deep Neural Networks (DNN) and put it into use in the world of gaming. We classified key presses on a given pre-labeled dataset to learn the game.
![Samples](images/img_1.png)

Inspired by OpenAI's paper on videogames-pre-training found [here](https://openai.com/research/vpt)


## Dataset
We took the labeled dataset at [Mario Dataset](https://github.com/rafaelcp/smbdataset) that has the following key-action mapping:
![img.png](images/img.png)

The dataset is divided in the following form: 

```bash
Folder: <user>_<sessid>_e<episode>_<world>-<level>_<outcome>

Frame: <user>_<sessid>_e<episode>_<world>-<level>_f<frame>_a<action>_<datetime>.<outcome>.png
```
where action is a number that needs to be converted into 8bits when each bit represents a key.

## Classifier
### Training
We trained a classifier based on ResNet18 architecture with BCE loss .The following set of parameters was used in the training process:

| Parameter             | Value |
|-----------------------|-------|
| Batch size            | 128   |
| Number of epochs      | 7     |
| Group size            | 15    |
| Initial learning rate | 1e-3  |
| use_color             | False |
| augmentations         | True  |


* group_size: amount of frames used for a single classification, not causal.
* augmentations: using invertion, gaussian blur, box blur, rotation and erasing from Kornia.

The training uses this set of buttons only: {Up, Left, Right, B}

### How to train

download the dataset from [Mario Dataset](https://github.com/rafaelcp/smbdataset) and extract it to 
``` bash
./mario_dataset
```
We have the following file which you can change the model's path and the parameters listed in the table above.

``` bash
python: main_train.py
```


### Labeling New Video

![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/dp_compare.JPG)
### Based on the paper "Combining Sketch and Tone for Pencil Drawing Production" by Cewu Lu, Li Xu, Jiaya Jia
#### International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June 2012
Project site can be found here:
http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm

Paper PDF - http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/npar12_pencil.pdf

Draws inspiration from the Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing

In this notebook, we will explain and implement the algorithm described in the paper. This is what we are trying to achieve:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/ExampleResult.JPG)

We can divide the workflow into 2 main steps:
1. Pencil stroke generation (captures the general strucure of the scene)
2. Pencil tone drawing (captures shapes shadows and shading)

Combining the results from these steps should yield the desired result. The workflow can be depicted as follows:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/Workflow.JPG)

* Both figures were taken from the original paper

Another example:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/jl_compare.JPG)

# Usage
```python
from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
ex_img = io.imread('./inputs/11--128.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                       stroke_darkness= 2,tone_darkness=1.5)
plt.rcParams['figure.figsize'] = [16,10]
plt.imshow(ex_im_pen)
plt.axis("off")
```
# Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above

# Folders
* inputs: test images from the publishers' website: http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm
* pencils: pencil textures for generating the Pencil Texture Map


# Reference
[1] Baker, Bowen, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. 2022. Video pretraining (vpt): learning to act by watching unlabeled online videos. arXiv: 2206.11795 [cs.LG].

[2] Pinto, R.C. 2021. Super mario bros. gameplay dataset. https://github.com/raf aelcp/smbdataset.

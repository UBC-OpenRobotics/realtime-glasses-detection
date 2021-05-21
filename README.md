# Real-time Glasses Detection

## UBC OpenRobotics

This repository is a modified version of the realtime-glasses-detection repository [here](https://github.com/TianxingWu/realtime-glasses-detection).

## Requirements
* python 3.6
- numpy 1.14
* opencv-python 3.4.0
- dlib 19.7.0

## Usage

The main script was altered to accomodate integration, and is now called `glasses_detector.py` and can be run as follows
```
python3 glasses_detector.py -i test/ -s
```
The `-i` flag accepts a path to an image or a folder of images, and `-s` displays the results visually.

The endpoint of the script is the `results` dictionary whose keys are indices corresponding to each face found in frame (which shouldn't be necessary given that cropped images of individuals should be fed in) and corresponding values are the face bounding box and binary classification of glasses/no-glasses

```
{0: {'face_bbox': [67, 67, 129, 129], 'class': True}}
```

## Reference
Jiang, X., Binkert, M., Achermann, B. et al. Pattern Analysis & Applications (2000) 3: 9. https://doi.org/10.1007/s100440050002

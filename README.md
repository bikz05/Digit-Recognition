# Digit Recognition
Handwritten Digit Recognition using OpenCV, sklearn and Python | [Video](https://www.youtube.com/watch?v=ur6JY2Hl-MM)

**Check out the blog post [here](http://hanzratech.in/python/handwritten-digit-recognition-using-opencv-sklearn-and-python/) for complete notes on how the code works.**

# Dependencies
1. `cv2`
2. `sklearn`
3. `skimage`
4. `numpy`
5. `collections`

# Contents
This repository contains the following files-

1. `generateClassifier.py` - Python Script to create the classifier file `digits_cls.pkl`.
2. `performRecognition.py` - Python Script to test the classifier.
3. `digits_cls.pkl` - Classifier file for digit recognition.
4. `photo_1.jpg` - Test image number 1 to test the classifier
5. `photo_2.jpg` - Test image numbre 2 to test the classifier

## Usage 

* Clone the repository - 
```bash
cd 
git clone https://github.com/bikz05/digit-recognition.git
cd digit-recognition
```
* The next step is to train the classifier. To do so run the script `generateClassifier.py`. It will produce the classifier named `digits_cls.pkl`. 

**NOTE** - *Already created `digits_cls.pkl` is done using python2, please run generateClassifier.py to generate pkl file*
```python
python generateClassifier.py
```
* To test the classifier, run the `performRecognition.py` script.
```python
python performRecognition.py -c <path to classifier file> -i <path to test image>
```
ex -
```python
python performRecognition.py -c digits_cls.pkl -i photo_1.jpg
```

## Results

### Sample Image 1
![Result Number 1](http://hanzratech.in/figures/digit-reco-1-out.png)
### Sample Image 2
![Result Number 2](http://hanzratech.in/figures/digit-reco-2.png)

## TODO

* Add a CNN Based approach
* Reject bounding boxes lesser than some area
* Look into user errors

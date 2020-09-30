# Facial expressions detection
## This project detects facial expressions of humans using Deep Learning and Computer Vision

### Here's a snippet

![expression_detection](https://user-images.githubusercontent.com/61016383/94587769-afa85680-02a0-11eb-897f-ffe8d88becbe.gif)

## About Project
  
  - #### Using [TensorFlow](https://www.tensorflow.org/), I have created a deep learning ConvNet Model `detection_model.h5` to detect Happy, Sad, Angry and Surprised expressions.
  
  
  - #### The model is trained on the [FER-2013 Kaggle Dataset](https://www.kaggle.com/msambare/fer2013).
  
  
  - #### You can find the model building and training process in the `model_building.ipynb` file.
  
  
  - #### Using OpenCV and Python, I have extracted the facial features with the help of Haar Cascade Classifier `haarcascade_frontalface_default.xml`, and given them to my model to detect the expressions.
  
  
  - #### I have written two Python scripts viz. `detection_on_vid.py` and `detection_on_image.py`. The former detects facial expressions using Webcam and the latter detects the    expressions of human faces in an image. 
  
  
## Libraries Used

  - #### TensorFlow 2.0
  
  - #### Keras
  
  - #### OpenCV
  
  - #### Python>=3.7
  
  - #### Numpy
  
  
## Here's what you can do to use this project to detect your expressions on the webcam:
Make sure you have Python>=3.7 installed in your machine, if not then you can download it [here](https://www.python.org/).

  - [Clone](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository) this repo or download the zip folder.
  
  - Install the following dependencies:
    
    - TensorFlow 2.0 `pip install tensorflow`
    
    - OpenCV `pip install opencv-python`
    
  - Go to the directory of this repo cloned or downloaded onto your local machine and open cmd.
  
  - Run `detection_on_vid.py` file.


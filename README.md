# Deep-Learning-with-OpenCV
Image recognition with OpenCV

Using OpenCV DNN Module for image recognition. 

Model supported in DNN: 
- Image Classification: Alexnet, GoogLeNet, VGG, DenseNet
- Object Detection: MobileNetSSD, VGG SSD, Faster R-CNN, Efficient Det
- Image Segmentation: DeepLab, UNet, FCN, OpenCV FaceDetector
- Text detection and recognition: Easy OCR, CRNN
- Human Pose estimation: Open Pose, Alpha Pose
- Person and face detection: Open face, Torchreid, Mobile FaceNet

Frameworks supported by DNN: 
- Darknet
- ONNX
- PyTorch
- TensorFlow
- Caffe

DNN Inference Process: 
1. Load the model
2. load the image
3. classify the loaded images with the model
4. display the inference

In this demo, you can upload images from local or from the 'url'. 

Run the program: 
This demo utilizes the streamlit to create the live web demo. Go to the terminal, type: 
streamlit run Image_classficiation_app.py

Python package: 
- streamlit
- opencv-python-headless
- opencv-python
- numpy
- pillow
- requests

Classification classes: classification_classes_animals.txt  #for animals, birds recognition
model: caffe _ DenseNet

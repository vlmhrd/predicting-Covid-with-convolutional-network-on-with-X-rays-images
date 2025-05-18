# predicting-Covid-with-convolutional-network-on-with-X-rays-images

# Goal
The goal of this project is to build and train a convolutional neural network (CNN) to classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, or Normal. This supports rapid and automated diagnosis using medical imaging.

# Steps
1. Import Libraries

TensorFlow, Keras, NumPy, scikit-learn, and Matplotlib are imported to handle deep learning, data processing, metrics, and visualization.

2. Data Preparation
   
The dataset is loaded from directories using Keras' ImageDataGenerator.
Images are rescaled and augmented (zoom, rotation, shifting) for the training set.
Training and validation iterators are created for model input.

3. Model Design
   
A Sequential CNN model is constructed with:
- Input layer for 256x256 grayscale images
- Two convolutional and max pooling layers, with dropout for regularization
- Flattening and a dense output layer with softmax activation for 3-way classification
- The model is compiled with the Adam optimizer and categorical cross-entropy loss.
  
4. Training

The model is trained on the X-ray images.
Early stopping is implemented to prevent overfitting, monitoring validation AUC.

5. Evaluation

A confusion matrix is computed to analyze the model's performance on the validation set.
Key metrics like accuracy and AUC are tracked during training.

6. Visualization

Training and validation accuracy, as well as AUC, are plotted over epochs to visualize learning progress.
# Tools and Libraries Used
- TensorFlow & Keras: For building and training the deep learning model.
- scikit-learn: For metrics such as the confusion matrix.
- Matplotlib: For plotting training metrics and results.
- NumPy: For numerical operations.
- Kaggle Dataset: COVID-19 X-ray image dataset (referenced at: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

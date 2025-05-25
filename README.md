# predicting-Covid-with-convolutional-network-on-with-X-rays-images

# Goal
The goal of this project is to build and train a convolutional neural network (CNN) to classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, or Normal. This supports rapid and automated diagnosis using medical imaging.

# Tools and Libraries Used
- TensorFlow & Keras: For building and training the deep learning model.
- scikit-learn: For metrics such as the confusion matrix.
- Matplotlib: For plotting training metrics and results.
- NumPy: For numerical operations.
- Kaggle Dataset: COVID-19 X-ray image dataset (referenced at: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

# Conclusion 

- The model achieved high categorical accuracy and AUC scores on both the training and validation sets (as indicated by the training logs and performance plots).
- The confusion matrix shows that the model can reasonably distinguish between COVID-19, Viral Pneumonia, and Normal cases, although there are some misclassifications.
- The project demonstrates that a relatively simple CNN architecture, when trained with proper preprocessing and data augmentation, can be effective in medical image classification for COVID-19 detection.

In summary, the project successfully applies deep learning to classify chest X-rays for COVID-19 and related conditions, showing the potential of AI-assisted diagnosis in medical imaging.

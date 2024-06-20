# README

## Project Overview

This project focuses on predicting potato plant diseases using image data. Utilizing the Potato Disease dataset, we implemented and evaluated several convolutional neural network (CNN) models, including a simple CNN, a model with added dropout layers, and a model with additional convolutional and dense layers. The primary objectives were to:

1. Preprocess the dataset to prepare it for model training.
2. Train and evaluate multiple CNN architectures.
3. Identify the model with the highest accuracy and interpret its key findings.
4. Save the trained models for future use.

## Dataset
Dataset souce - PlantVillage 
The dataset used for this project consists of images of potato plant leaves, categorized into three classes:
- Early Blight
- Late Blight
- Healthy

The dataset was split into training and validation sets to evaluate the performance of the models.

## Key Findings

- The baseline CNN model achieved a training accuracy of approximately 95.98% and a validation accuracy of 95.80%.
- The CNN model with dropout layers achieved a training accuracy of approximately 92.14% and a validation accuracy of 95.90%.
- The CNN model with additional layers achieved a training accuracy of approximately 98.93% and a validation accuracy of 96.90%.

## Instructions for Running the Notebook and Loading the Saved Models

### Prerequisites

1. Python 3.8 or higher
2. Jupyter Notebook or JupyterLab
3. The following Python packages:
   - numpy
   - pandas
   - matplotlib
   - seaborn
   - scikit-learn
   - tensorflow
   - keras

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Loading the Saved Models
The models have been saved in the models directory. To load a saved model and make predictions, use the following code snippet in a new Jupyter Notebook cell:

```python

from keras.models import load_model

# load the desired model
model_path = 'saved_models/model_1.h5'
model = load_model(model_path)

# example usage: Predicting the class of a new image
import numpy as np
from keras.preprocessing import image

# load and preprocess the image
img_path = 'data/images/sample_1.jpg'  
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

# map the predicted class index to the class label
class_labels = ['Early Blight', 'Late Blight', 'Healthy']
print(f'Predicted Class: {class_labels[predicted_class[0]]}')
```


# Optimizations and Parameter Setting: Discussion of Results
### Objective

The primary objective of this project is to accurately classify potato plant images into three categories: Early Blight, Late Blight, and Healthy. To achieve this, several optimizations and parameter settings were employed to enhance the model's performance.

### Dataset

The dataset used comprises 2152 images of potato leaves, categorized into three classes:

Early Blight: 1000 images
Late Blight: 1000 images
Healthy: 152 images

### Model Architecture

The model architecture selected is a Convolutional Neural Network (CNN), which is well-suited for image classification tasks. The initial architecture included:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with Dropout for regularization
- Softmax output layer
  
## Key Findings from Parameter Tuning and Optimizations

### Learning Rate Optimization:

- Initial Learning Rate: Started with a default learning rate of 0.001.
- Optimization: Experimented with different learning rates (0.01, 0.001, 0.0001). The optimal learning rate was found to be 0.0001, which balanced the training speed and accuracy without causing the model to overfit.
- 
- Batch Size:

Initial Batch Size: Started with a batch size of 32.
- Optimization: Tested batch sizes of 16, 32, and 64. A batch size of 32 provided the best balance between computational efficiency and model performance.
  
- Number of Epochs:

Initial Epochs: Initially trained for 10 epochs.
Optimization: Extended to 50 epochs. Early stopping was used to monitor validation loss, which helped in preventing overfitting. The optimal number of epochs was found to be around 30, where the validation loss started to stabilize.

- Dropout Rate:

Initial Dropout Rate: Started with a dropout rate of 0.5.
Optimization: Tested dropout rates of 0.3, 0.5, and 0.7. A dropout rate of 0.5 provided the best regularization effect, reducing overfitting without significantly impacting the model's ability to learn.

- Data Augmentation:

Techniques Used: Applied random rotations, flips, and shifts to increase the diversity of the training dataset.
Impact: Improved the model's robustness and generalization by providing more varied training examples. This led to a noticeable improvement in validation accuracy.

- Optimizer Choice:

Initial Optimizer: Adam optimizer was used initially due to its adaptive learning rate capabilities.
Optimization: Compared Adam with SGD and RMSprop. Adam remained the best performer in terms of convergence speed and final accuracy.
Final Model Performance

After implementing the above optimizations and parameter settings, the final model performance improved significantly. The key performance metrics are as follows:

Training Accuracy: 98.5%
Validation Accuracy: 96.2%
Test Accuracy: 95.8%

### Conclusion

The optimizations and parameter tuning significantly enhanced the model's performance, particularly in terms of generalization to unseen data. Key strategies included adjusting the learning rate, employing appropriate batch sizes, extending the number of epochs with early stopping, utilizing dropout for regularization, and applying data augmentation techniques.

These steps collectively contributed to achieving a robust and accurate model for potato disease classification. In the future work, I could explore more advanced architectures like transfer learning with pre-trained models (e.g., ResNet, VGG) to potentially further improve accuracy and efficiency.

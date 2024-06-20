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

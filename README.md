# Bird Classification Model using EfficientNetB0

# Overview
This project involves building a bird classification model using EfficientNetB0, which is a pre-trained deep learning model from TensorFlow. The model was trained on a dataset of bird images to classify them into 525 different species.

#  Project Structure
* Data Preprocessing: The images were split into training, validation, and test sets using ImageDataGenerator. The EfficientNetB0 model was used to preprocess the images.
* Model Architecture: EfficientNetB0 was used as the base model, and custom layers were added on top for fine-tuning and classification. The model's weights were initialized using the ImageNet dataset, and additional layers were added for this specific task.
* Callbacks: Several callbacks were used to improve training, including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.

# Libraries Used
- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pandas
- Seaborn
- OpenCV
- scikit-learn
- Keras Preprocessing

# Model Workflow
1. Data Preprocessing:
The ImageDataGenerator class was used to preprocess the images and split the data into training, validation, and test sets:

* Training Data: 80% of the data.
* Validation Data: 20% of the training set.
* Test Data: 20% of the total dataset.
2. Model Architecture:
The EfficientNetB0 model was loaded without the top layer, and custom fully connected layers were added
3. Model Training:
The model was compiled and trained using the Adam optimizer with a learning rate of 0.0001

# Callbacks:
* ModelCheckpoint: Saves the best model based on validation accuracy.
* EarlyStopping: Stops training when validation loss doesn’t improve for 5 epochs.
* ReduceLROnPlateau: Reduces learning rate if validation loss doesn’t improve after 3 epochs.
  
4. Results:
After training for 50 epochs, the model achieved the following results on the test dataset:

* Test Loss: 0.52141
* Test Accuracy: 87.03%

# Conclusion
This bird classification model achieved a Test Accuracy of 87.03%, demonstrating strong performance in classifying birds across 525 species. The model was optimized using techniques like dropout, callbacks, and learning rate adjustment.

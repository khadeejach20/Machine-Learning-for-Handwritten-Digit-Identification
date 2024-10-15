# Machine Learning for Handwritten Digit Identification

## Overview

This project focuses on classifying handwritten digits using the MNIST dataset. The goal is to develop a machine learning model that can identify handwritten digits effectively while minimizing overfitting and underfitting through regularization techniques.

## Dataset

The MNIST dataset consists of 70,000 images of handwritten digits (0-9), with 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image.

## Project Structure

- `ipynb` (Jupyter Notebook) file for model development.
- Various versions of models using different regularization techniques.
- Visualizations of model performance over epochs.

## Model Architecture

The project explores different model architectures and regularization techniques, including:

1. **Baseline Model**:
   - Dense layer with 128 units and ReLU activation.
   - Dropout layer with a dropout rate of 0.5.
   - Output layer with 10 units (softmax activation).
   - Achieved a test accuracy of approximately **97.23%**.

2. **Regularized Model**:
   - Increased complexity with additional dense layers (512, 256, and 128 units).
   - Dropout layers after each dense layer.
   - L2 regularization applied to kernel weights.
   - Achieved a test accuracy of approximately **96.74%**, which indicated potential underfitting due to increased regularization.

3. **Final Model**:
   - Similar architecture to the regularized model but without L2 regularization.
   - Maintained dropout layers.
   - Increased epochs from 10 to 13.
   - Achieved a test accuracy of approximately **97.93%**, demonstrating effective learning and generalization.

## Training

The models are trained using the `rmsprop` optimizer with categorical crossentropy loss. The training process includes the following steps:

- Train the model on the training set with a validation split to monitor overfitting.
- Evaluate the model's performance on the test set.

## Performance Evaluation

### Training and Validation Accuracy

The training and validation accuracy consistently improved over epochs, indicating effective learning and generalization:

- Baseline Model: Achieved a final test accuracy of **97.23%**.
- Regularized Model: Achieved a final test accuracy of **96.74%**, suggesting potential underfitting.
- Final Model: Achieved a final test accuracy of **97.93%**, indicating improved performance.

### Loss Values

- The reduction in training and validation losses indicates effective minimization of errors across epochs.

### Visualization

The performance of the models is visualized through plots of training and validation accuracy and loss over epochs.

## Conclusion

Through various modifications and regularization techniques, the models demonstrated the ability to generalize well to unseen data. The final model, which balanced dropout regularization and model complexity, achieved the highest test accuracy, confirming the effectiveness of these strategies in preventing overfitting.

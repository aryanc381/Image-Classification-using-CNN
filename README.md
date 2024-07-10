# Image Classification of Images using CNN Model to Decipher Between Cat or Dog

This project aims to classify images as either a cat or a dog using a Convolutional Neural Network (CNN). The model is trained and evaluated on a dataset of cat and dog images.

## Project Structure

- `catvdog.ipynb`: Jupyter Notebook containing the implementation of the CNN model for image classification.

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

## Dataset

The dataset used in this project consists of images of cats and dogs. You can download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). The dataset should be structured as follows:

```
dataset/
    training_set/
        cats/
            cat1.jpg
            cat2.jpg
            ...
        dogs/
            dog1.jpg
            dog2.jpg
            ...
    test_set/
        cats/
            cat1.jpg
            cat2.jpg
            ...
        dogs/
            dog1.jpg
            dog2.jpg
            ...
```

## Model Architecture

The CNN model used in this project has the following architecture:

1. Convolutional Layer with 32 filters, kernel size of 3x3, ReLU activation, and input shape (64, 64, 3).
2. Max Pooling Layer with pool size of 2x2.
3. Convolutional Layer with 64 filters, kernel size of 3x3, and ReLU activation.
4. Max Pooling Layer with pool size of 2x2.
5. Convolutional Layer with 128 filters, kernel size of 3x3, and ReLU activation.
6. Max Pooling Layer with pool size of 2x2.
7. Flatten Layer.
8. Fully Connected Layer with 128 units and ReLU activation.
9. Output Layer with 1 unit and Sigmoid activation.

## Training

The model is trained using the following configuration:

- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Batch Size: 32
- Number of Epochs: 25

The model is trained on the training set and validated on the validation set. During training, the training and validation accuracy and loss are monitored to avoid overfitting.

## Evaluation

The model is evaluated on a separate test set to measure its performance. The accuracy and loss are plotted to visualize the training process. The confusion matrix is also computed to provide insight into the model's performance.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repository.git
```

2. Navigate to the project directory:

```bash
cd your-repository
```

3. Install the required packages:

```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

4. Download and extract the dataset as described in the [Dataset](#dataset) section.

5. Open the Jupyter Notebook:

```bash
jupyter notebook catvdog.ipynb
```

6. Run the cells in the notebook to train and evaluate the model.

## Results

The model achieves an accuracy of approximately X% on the test set. The training and validation accuracy and loss are plotted as follows:

![Training and Validation Accuracy](path/to/accuracy_plot.png)
![Training and Validation Loss](path/to/loss_plot.png)

The confusion matrix for the test set is as follows:

```
Confusion Matrix:
[[TN  FP]
 [FN  TP]]
```

## Conclusion

This project demonstrates how to build and train a CNN model for binary image classification. The model successfully classifies images of cats and dogs with a reasonable accuracy. Future improvements could include data augmentation, hyperparameter tuning, and exploring different model architectures.


## Acknowledgments

- The dataset was provided by Kaggle.
- The project was inspired by various online tutorials and resources on CNNs and image classification.
```

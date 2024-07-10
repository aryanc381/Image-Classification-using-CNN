# Image Classification using CNN Model to classify two objects (in this case - cats and dogs)
<p align="center"> <img src='https://github.com/aryanc381/Image-Classification-using-CNN/blob/main/catdog.jpg',alt="catvdog" width='1550' height='450'/></p>

In this project I have used a CNN model with mainly three filters with (32, 64, 128) nodes respectively to categorise two objects in different classes based on their features. The accuracy I had achieved as of ```10th-July-2024``` was 98.1% with 10 epochs of training on a v4-GPU on google-colab.
## Project Structure 
- ```catvdog.ipynb``` - This is the main jupyter-notebook that contains the implementation of the CNN Model along with the pre-processing, normalization and finally, the prediction.
- ```README.md``` - Contains the concept behind the project and references.
- ```test file```- Consists images that are used for testing the model.
- ```train file``` - Consists images that are used for training the model.

## Requirements 
- Python
- Code-Editor (JupyterNB, VSCode, etc)
- Tensorflow
- Keras
- Matplotlib
- OpenCV
- Scikit
You can install the required libraries using the following commands in python :
```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

## Dataset
Find the entire ```1.0GB``` dataset here : [training and testing dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## Model Architecture 
The CNN Model has the following architecture :
1. Convolutional Layer with 32 filters, kernel size of 3x3, ReLU Activation, and input shape (64, 64, 3).
2. Max-Pooling Layer with pool size of 2x2.
3. Convolutional Layer with 64 filters, kernel size of 3x3, and ReLU Activation.
4. Max-Pooling Layer with the size of 2x2.
5. Convolutional Layer with 128 filters, kernel size of 3x3, ReLU Activation.
6. Max-Pooling Layer with the size of 2x2.
7. Flatten Layer.
8. Fully connected layers with 128 units and ReLU Activation.
9. Output Layer with 1 unit of Sigmoid Activation.

## Concept
### Convolutional Layer
#### 1. **A basic understanding :**
 - This layer is the fundamental building block of a neural network which uses grid-layering to process image data.
 - The main purpose of this layer is to extract and automatically understand spatial hierarchial features from the input image.
 - **Filters** - These are small learnable weights or matrices that slide over the image to form feature maps.
 - **Strides** - This is the step size with which the filter moves across the image, for example, in a hypothetical situation a stride with one step size will move with one pixel at a time and stride ith 2 step sizes will move two pixels at a time.
 - **Padding** - Sometimes the filters do not fit the size of the image dimensions which is where padding is used. Padding is basically a function in convolutional layer that adds pixels to the boundary of the image to fit the filter in the image, examples are ```valid``` (no padding) OR     ```same``` (padding to keep o/p size same as i/p size).
#### 2. **How it works :**
  - **Convolutional Operation** - Each filter slides across the input image. At each position, elementwise matrix multiplication is carried out b/w the filter and the image that is covered. Finally, the summation of those values results into a feature map.
  - **Activation** - After Convolution, an activation is added to add non-linearity to the neural network that is important to understand complex computations, ofcourse ML is maths after all and if this was all linear (which by the way is not the right method to understand patterns in an image because ofcourse images cant be linear every time) might as well call it a 'fancy linear-regression model' haha xD.
<p align="center"><img src="https://github.com/aryanc381/Image-Classification-using-CNN/blob/main/conv_layer.jpeg", alt="convlayer" width="700" height="300"/></p>

- **Pooling** - Pooling which is an operation that is optional while developing a neural network is used to reduce the spatial dimensions of the feature map. This helps in reducing the computational load and also helps in reducing overfitting of data (This is something that I've learnt based on the projects I have made, not a declaration, it may work out differently for different data). Common pooling operations are ```Max-Pooling``` (taking the maximum value) and ```Min-Pooling``` (taking the minimum value).

### MaxPooling Layer
#### 1. **A basic understanding :**
- A Max-Pooling layer is a type of down-sampling operation often used in CNN which reduces the dimensionality of the input hile retaining the important features of the image.
- Spatial Dimensions are majorly reduced i.e the width and the height of the input volume thereby reducing the parameters and the overall computation of the neural network.
- ```Translation Invariance``` is one of the important aspects of MaxPooling Layer where the algorithm broadly decides which part of the image has a larger impact on the accuracy thereby considering it and neglecting the part where the matrix of the image has a very minimal impact.
- Components of a pooling layer are same to that of a convolutional layer - Pooling window, stride and padding.
#### 2. **How it works :**
- Similar to a convolutional layer, the pooling windows slides over the image i.e the input feature map created by the convolutional layer at each position and takes the maximum value from that region covered by the window.
- This value is assigned to the corresponding position in the output feature map.
- The main advantage o MaxPooling Layer is that it reduces overfitting while keeping the most important features and neglecting the garbage features of an image.

### Functions

#### **1. ReLU Activation :**
- I have used ReLU the most in a neural network.
- for x >= 0 return x AND for x < 0 return 0.
- This activation introduces non-linearity which is required to analyse complex patterns and sparse activations are seen as it outputs zero for any value that is less than zero.
- This makes ReLU better for computationality as a threshold is introduced i.e zero.
- (You will need to do more research to understand ReLU, its sort of vast but the pointers I have provided are enough for the notebook but yet I would recommend you to explore youtube or any resource available to understand ReLU).
- Reference : [100 days of ML](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)

<p align='center'><img src="https://github.com/aryanc381/Image-Classification-using-CNN/blob/main/relu_and_sigmoid.png", alt="sig_and_relu" width="600" height="200"/></p>

#### **2. Sigmoid Activation :**
- I have used sigmoid function as the problem statement has binary classification of categories (in this case : cats and dogs).
- Sigmoid function mainly represents a probability distribution working in the form of a smooth curve for binary classification.
- Hence this was the perfect output layer for this application
- I suggest you dive deep in understanding these activation functions that you can find in this playlist : [100 days of ML](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)

#### **3. Adam Optimizer :**
-  In this CNN model, I used Adam Optimizer to achieve global minimum - as simple as that.
-  It worked out well to be honest for binary classification. 

## Training
The model was trained using the following configuration :
- Loss Function : Binary Cross Entropy.
- Optimizer : Adam for gradient descent.
- Metrics : Accuracy.
- Batch Size : 32
- Number of epochs : 10
The model is trained on the training dataset and cross-validated on the validation dataset. During training, I have mapped the training and validation accuracy along with losses to reduce overfitting.

## Evaluation and Result
The model evaluation concluded that a minimum of ```83.20%``` of accuracy is required for somewhat-accurate (for the images with a lot of noise) to accurate classification of image between dog and cat. 

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
4. Finally, download and extract the dataset as described in the Dataset section.

5. Open the Jupyter Notebook:
```bash
jupyter notebook catvdog.ipynb
```
6. Run the cells in the notebook to train and evaluate the model.

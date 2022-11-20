# Convolutional Neural Network from scratch [CPP]

### See also: [Convolutional Neural Network from scratch [PY]](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY)
 
**Author: Marco Sento**

This project is a C++ framework that gives the possibility to generate istances of different layers typologies, in order to set-up your own neural network.  
The interface is similar to the one of other popular software such as PyTorch and Tensorflow, but this is not intended as an alternative to them rather as a personal challenge to acquire a deeper knowledge about the structure and working principles of Convolutional Neural Networks.

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/Screenshots/structure.png)

<h2>  Overview: </h2>

#### [Program structure](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/README.md#-program-structure-) 

#### [Build the network](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/README.md#-build-the-network-)  

#### [An example: MNIST classification](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY#-an-example-mnist-classification-)  

#### [Results](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/README.md#-results-)  


<h2> Program structure </h2>

The program is composed by 5 modules:

##### MLP.py:
Defines the basic block of the Multi Layer Perceptron and is in charge of generate the fully connected 	layers. 
The activation function implemented is the (old fashioned) sigmoid.  
The parameter update is performed using SGD or Adam.  
 
##### Filters.py: 
Defines the convolutional layers such as filters and pooling layers.  
The pooling layer perform a downsample of the input volume using a average pooling or a max pooling.  
The convolutional layers are initialized with random values and with user defined hyperparameters.  
They leverages on the Leaky ReLu activation function. They can also implement padding.  

##### CNN.py:
Defines the relations between the instantiated objects, it iterates through different layers the forward and backwards pass and manages the training, evaluation, testing and plotting steps.  

##### Datasets.py: 
It’s in charge of generate a compatible set of lists of datasets for the CNN.  
It implements different classes for each dataset available.  
The aim is also to performs normalization and zero-cenetering on the input.  

##### main.py:
To pick the wanted dataset and instantiate the user defined network.  


<h2> Build the network </h2>

To start building the neural network it’s enough to chain the desired types of layers.  
At first are used convolutional layers, that takes as input a volume of the size of the input images.  
Pooling layers can be added, those will perform a downsampling along the spatial dimensions.  
The network is concluded with the fully-connected layers that will compute the class scores.  

Then the desired dataset it’s loaded by calling the corresponding class from the “Datasets” module.  
At the moment, only the MNIST dataset has been implemented.  

Finally, just call up the functions for training and testing and print out the results.  
The preview frequency allows to preview feature maps.  

<h2> An example: MNIST classification </h2>

#### Dataset:
The image recognition task on the MNIST dataset is carried out using the above architecture.  
The dataset is composed of small images arranged in two sets, consisting of 60000 training pictures and 10000 for the test.  
The training dataset is then subdivided into 50000 samples for training and 10000 for validation.  

#### Network:
In the conv layers are used small filters, 3x3, using a stride of 1, and padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input.   
The convolutional layers are followed by Leaky ReLU activation functions.  
It’s used a Leaky ReLU to avoid that a large gradient backpropagating in the neuron deactivate it for all the training process.  
The pool layers are in charge of downsampling the spatial dimensions of the input, using a max-pooling with 2x2 receptive fields and with a stride of 2.    
This will discard exactly 75% of the activations in the input volume.  
Fully Connected layers (FC) use the sigmoid activation function, although it is an uncommon choice to mix different activations into one CNN, this choice let the output scores be considered as a probability distribution without using the Softmax classifier and the cross entropy loss.  

#### Loss function:
There are many ways to quantify the data loss, but in this example it’s used the Mean Squared Error (MSE), perhaps the simplest and most common loss function.  
The full loss function normally is defined as the average data loss over the training examples and the regularization.  
The regularization has not been introduced in this solution even if without it, the weights are free to get any arbitrary value among different training.  
This is not a desirable behaviour since this leads to a reduced ability to generalize: it is much better to get a classificiation value as sum of diffuse contributes of small weights rather than from one single product with a large weight.  
The effect of the regularization is then to reduce the absolute values of the weigth and spread it uniformly over different weights.  
This leads also to a gradual “forgetting” effects of the network that in turn reduce the overfitting.  
 It is then predictable that introducing this term in the final loss function would give an improvement of some percentage points.  

#### Parameter update:
The gradient from the backpropagation is used for the parameter update. Even though a minibatch gradient descent would be more computational efficient, has been implemented the stochastic GD, evaluating the gradient after each sample.  
Since it is usually helpful to anneal the learning rate over time, it has been applied an exponential decay.  
Two different possibilities have been defined for the parameter update: vanilla SGD and Adam.  
Applying the Adam per-parameter optimizer to the FC layers leads to a huge improvement of the performances as shown below.  

#### Babysitting the learning process:
Two possibilities are provided to verify the proper functioning of the network:  
  
The first is to perform a sanity check, before starting the training process, through the special function of the CNN.py module.  
In this way the net is trained on a subset of about 20 / 50 samples for many epoch. If everything works as it should then the Loss go to zero. 

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/Screenshots/sanity%20check.png)
  
The second debugging tool is to look at the feature maps produced by the different filters.  
These results are displayed with a user-defined period, expressed as a percentage of the dataset samples. This parameter is the preview_ratio.  
Below can be seen an example produced by the instantiated network.  

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/Screenshots/feature_maps.png)
  
<h2> Results </h2>

After the training it is possible to analyze the results obtained for both SGD and Adam.  
The loss function is quite variable as expected in the case of unit batch sizes but still a sufficiently low value has been reached.  
Using Adam the validation accuracy is equal to 91.16% while the test accuracy is 91.06%.  
The main difference is in the speed with which Adam achieves a good accuracy, resulting in a higher training accuracy. Accuracy shows no signs of overfitting. 

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/Screenshots/SDG%20results.png)

`FIG1 shows loss and accuracy for SGD.`

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/Screenshots/ADAM%20results.png)

`FIG2 shows loss and accuracy with Adam optimizer.`




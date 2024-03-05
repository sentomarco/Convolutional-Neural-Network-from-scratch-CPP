# Convolutional Neural Network from scratch [CPP]

### See also: [Convolutional Neural Network from scratch [PY]](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY)
 
**Author: Marco Sento**

This project is a C++ framework that gives the possibility to generate istances of different layers typologies, in order to set-up your own neural network.   
The interface is similar to the one of other popular software such as PyTorch and Tensorflow but this is just a personal challenge to acquire a deeper knowledge about the structure and working principles of Convolutional Neural Networks.

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-CPP/blob/main/Screenshots/structure.png)

<h2>  Overview: </h2>

#### [Program structure](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/README.md#-program-structure-) 

#### [Build the network](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/README.md#-build-the-network-)  

#### [An example: MNIST classification](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY#-an-example-mnist-classification-)  

#### [Results](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-PY/blob/main/README.md#-results-)  


<h2> Program structure </h2>

The program is composed by 6 modules:

##### MLP:
Defines the basic block of the Multi Layer Perceptron and is in charge of generate the fully connected 	layers. 
The activation function implemented is the (old fashioned) sigmoid.  
The parameter update is performed using SGD or Adam.  

##### Volumes: 
It allows to create tensors for store and manipulating images.  
It is a key element for the operation of the various modules, making image management very simple and fast.  
Images are treated as vectors but conceptually they are transposed into a n-dimensional volume and accessed as such, not unlike numpy.  
The number of dimensions is arbitrary.  

##### Filters: 
Defines the convolutional layers such as filters and pooling layers.  
The pooling layer perform a downsample of the input volume using a average pooling or a max pooling.  
The convolutional layers are initialized with random values and with user defined hyperparameters.  
They leverages on the Leaky ReLu activation function. They can also implement padding.  

##### CNN:
Defines the relations between the instantiated objects, it iterates through different layers the forward and backwards pass and manages the training, evaluation, testing and plotting steps.  

##### Datasets: 
It’s in charge of generate a compatible set of lists of datasets for the CNN.  
It implements different classes for each dataset available.  
The aim is also to performs normalization and zero-cenetering on the input.  

##### main:
To pick the wanted dataset and instantiate the user defined network.  


<h2> Build the network </h2>

To start building the neural network it’s enough to chain the desired types of layers.  
At first are used convolutional layers, that takes as input a volume of the size of the input images.  
Pooling layers are still to be implemented.  
The network is concluded with the fully-connected layers that will compute the class scores.  

Then the desired dataset it’s loaded by calling the corresponding class from the “Datasets” module.  
At the moment, only the MNIST dataset has been implemented.  

Finally, just call up the functions for training and testing and print out the results.  

#### About tensors:   
All the multidimentionals objects are Volumes: images, datasets and filters conceptually are all tensors and are manipulated as such.  
In practice the values are stored as vectors but the conversion to tensors is managed by indexing as follows:  
For a generic point p = [I,J,K] in a volume v = [H,W,D] the corresponding index in the vector is: i = I + J*H + K*W  

![image](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-CPP/blob/main/Screenshots/vector-tensor.png)

<h2> An example: MNIST classification </h2>

#### Dataset:
The image recognition task on the MNIST dataset is carried out using the above architecture.  
The dataset is composed of small images arranged in two sets, consisting of 60000 training pictures and 10000 for the test.  

#### Network:
In the conv layers are used small filters, 3x3, using a stride of 2 in order to perform a downsampling of the spatial dimensions of the input.
The convolutional layers are followed by Leaky ReLU activation functions.  
It’s used a Leaky ReLU to avoid that a large gradient backpropagating in the neuron deactivate it for all the training process.  
 
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

#### Babysitting the learning process:
In order to verify the proper functioning of the network it is possible to perform a sanity check, before starting the training process, through the special function of the CNN module.  
In this way the net is trained on a subset of about 20 / 50 samples for many epoch. If everything works as it should then the Loss go to zero. 

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-CPP/blob/main/Screenshots/check.png)
  
Another good debugging tool is to look at the feature maps produced by the different filters.  
These plots are obtained by means of a python script that extrapolates the feature maps produced by the network.  
Below can be seen an example produced by the instantiated network.  

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-CPP/blob/main/Screenshots/preview.png)
  
<h2> Results </h2>

After the training it is possible to analyze the results obtained.  
The loss function is quite variable as expected in the case of unit batch sizes but still a sufficiently low value has been reached.  
The overall test accuracy reached is almost 80%.  

It is interesting to note that, training a neural network in python requires 500% of the time it takes to train the same network implemented in C++.  
In 10 min. you train a network that in python required 50 min.

Below is also reported a graph obtained analyzing the collected data with a python script.  
The gap between the training and test accuracy indicates a certain amount of overfitting.  
There are several ways of prevent overfitting such as inplemeting the L2 regularization or using dropout.  
Taking advantage of pooling layers would also be effective against overfitting.  

![immagine](https://github.com/sentomarco/Convolutional-Neural-Network-from-scratch-CPP/blob/main/Screenshots/SDG%20results.png)

`FIG1: loss and accuracy for SGD.`





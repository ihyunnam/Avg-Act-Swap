# The Avg-Act Swap
The Avg-Act Swap ('the Swap') is a technique to place the average pooling layer ('AvgPool') before an activation function ('Activation') in machine learning models over encrypted data, unlike the convention in machine learning over unencryped data to place the layers the other way around. The Swap is designed to be deployed in machine learning models operating on data encrypted with fully homomorphic encryption. It speeds up encrypted inference by reducing the number of expensive homomorphic multiplications by a factor of k^2 (k=AvgPool kernel size).

## What's On This Page
This repository presents three neural networks using the Swap: a 5-layer CNN (5-swap), an 8-layer CNN (8-swap), and Lenet-5 modified to use the Swap. For performance comparison, we also include the equivalents of the three neural networks that do not utilize the Swap. In each neural network, three options for the activation function are given: the Tanh, Sigmoid, and Square functions.

## Experimental Results
All experiments were conducted on an e2-standard-32 machine with 32 vCPUs and 128GB of memory. The CPU platform is Intel Broadwell. Compared to the 8-layer model without the Swap, 5-swap acheived a 47% reduction in encrypted inference time and a 96% accuracy using Tanh activation with k=3, and a 35% reduction in encrypted inference time and a 98% accuracy using Square activation with k=3. The 8-layer model using the Square Activation achieved 

Furthermore, we modified Lenet-5 to use the Avg-Act Swap in the second occurrence of the model to achieve a 27% reduction in encrypted inference speed with a 90% accuracy.

More details will be available in the paper: (to be uploaded soon).

## Design
The models are trained with unencryped MNIST training images. The models are then made FHE-friendly through quantization and other measures to respect FHE constraints. The models are evaluated to classify MNIST test images encrypted using the Cheon-Kim-Kim-Song FHE scheme.

All implementations presented in this repository use PyCrCNN to implement neural network layers and Pyfhel to implement FHE.
PyCrCNN Source: https://github.com/AlexMV12/PyCrCNN
Pyfhel Source: https://github.com/ibarrond/Pyfhel

## Project Information
This research project was supported by a Stanford VPUE Major Grant for summer 2023 and used as a part of my CS Honors Thesis at Stanford University.

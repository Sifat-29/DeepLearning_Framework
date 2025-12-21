# A DeepLearning Framework from Scratch

This is a highly modular deep learning library built completely in pure C without any external dependencies. It implements the core mathematics of neural networks in the form of tensors, automatic differentiation and backpropagation thus allowing the user to build, train and run deep learning models
 from the ground up.

[**Link to project**](https://github.com/Sifat-29/DeepLearning_Framework/blob/main/README.md)

<img width="1310" height="688" alt="Screenshot 2025-12-22 040650" src="https://github.com/user-attachments/assets/2cf67ef2-05d7-47f3-a426-f19b0f8fd26a" />

## How It's Made:

**Tech used:** C (Standard C99), GCC, Makefile

Building this framework was intended an exercise in understanding the mathematics behind Deeplearning and how basic functionalitites of libraries like Tensorflow and pyTorch are implemented. Instead of using high-level APIs I engineered the low-level systems that make deep learning possible from scratch:

1.  **Tensor Engine:** A custom linear algebra engine. I implemented struct-based Tensors with dynamic memory allocation, handling matrix multiplication, transposition and element-wise operations.
2.  **Modular Architecture:** It is similar to the Keras-style API where a `Network` struct acts as a container for a dynamic array of `Layer` objects. This is to stack Dense layers with various activations (ReLU, Linear) easily.
3.  **Backpropagation:** I implemented the chain rule manually for fully connected layers. This involved calculating gradients for weights, biases, and inputs and caching the necessary intermediate values (Forward Pass Cache) to perform the Backward Pass correctly.
4.  **Optimisers:** I built a stateful SGD optimiser that handles parameter updates. This required decoupling the optimisation logic from the layer logic to allow for future enhancements in the form Momentum, Adam, etc.

## Optimisations

Some of the major optimisations I made:

*   **Memory Recycling:** In the training loop, intermediate tensors (predictions, gradients of hidden layers) are allocated and freed immediately within the cycle. I ensured zero memory leaks by carefully tracking pointer ownership, keeping the memory footprint minimal even for large datasets.
*   **In-Place Operations:** To reduce the overhead of `malloc`/`free`, I implemented in-place mathematical operations (e.g., `tensor_add_scaled_inplace`) for the optimizer steps, modifying weights directly in memory rather than creating new tensor copies.
*   **Matrix Multiplication Optimisation:** Transposed one of the matrix to execute the matrix multiplication so that both traversals are in row-major order. This improved cache locality and thus improved runtime by approximately 20%.
*   **Numerical Stability:** I implemented **He Initialisation** (`sqrt(6/n)`) for weights to solve the "Dying ReLU" problem, where gradients would vanish, and the network would stop learning.
*   **Mini-Batch Processing:** Initially, I trained using Stochastic Gradient Descent (Batch Size = 1). By refactoring the math to support Matrix-Matrix multiplication (Batch Size = 64), I drastically improved training speed and CPU cache utilisation.

## Problems Encountered and Lessons Learned:

*   **Intense Manual Memory Management:** Early versions of the training loop leaked memory rapidly because intermediate tensors (activations and gradients) were not being freed after backpropagation.

I implemented a strict memory ownership protocol where the Network struct retains ownership of the permanent parameters (Weights/Biases) whereas the training loop manages the lifecycle of the temporary tensors.

*   **The "Dying ReLU" Problem:** The first test of a neural network was done on the XOR problem with all the layers having RELU as their activation function. Upon training the network multiple times, I was able to learn the XOR ciruit only about half of the times. Upon researching about the network architecture I was using, I discovered that the neurons might die because of the random values initialised for the weights and gradients and the properties of the RELU function. 

Implementing He Initialisation, having positive biases initially  and having LINEAR as the visible layer dramatically fixed this issue.


# TODO

### Activations and Loss
* Implement Softmax coupled with CCE

### Optimiser
* Implement SGD with Momentum
* Implement Adam

### General features
* Implement Network saving/retrieving functionality

### Runtime Optimisation
* Optimise runtime even more by adding buffers to reduce malloc/free calls

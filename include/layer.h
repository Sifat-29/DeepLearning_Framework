#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "activations.h"



typedef struct Layer {

    int n_neurons;                    // Number of neurons in this layer
    int n_neurons_prev;               // Number of neurons in the previous layer to which this layer is densely connected

    Tensor* weights;                  // (n_neurons_prev x n_neurons)
    Tensor* biases;                   // (1 x n_neurons)

    Activation* activation;           // Activation function for this layer, is able to give activated tensor and gradient of activation
    
    Tensor* d_weights;                // Gradient of weights (Kept for the optimiser to optimise after a backward pass)
    Tensor* d_biases;                 // Gradient of biases  (Kept for the optimiser to optimise after a backward pass)

    Tensor* input_transpose_cache;    // Stores 'XT' 
    Tensor* z_cache;                  // Stores 'Z' = W @ X + B

} Layer;



// ==========================================
//             Object Management
// ==========================================

/**
 * Returns a new layer based on the input parameters
 * 
 * @param n_neurons Number of neurons in this layer.
 * @param prev_n_neurons Number of neurons in the prev layer (the layer connected to this).
 * @param act_func_name Name of the activation functions out of the available ones.
*/
Layer* create_layer(int n_neurons, int n_neurons_prev, activation_function func);



/**
 * Completely frees the layer
 * 
*/
void free_layer(Layer** layer);



// ==========================================
//          Training and Prediction
// ==========================================

/**
 * Returns the output of the forward pass performed on the layer with a given input.
 * Returns NULL if fails.
 * 
 * @param layer The layer on which the forward pass is performed
 * @param input The input tensor (batch_size x n_neurons_prev) on which the forward pass is performed
*/
Tensor* forward_pass(Layer* layer, Tensor* input);



/**
 * Returns the gradient of output this layer so it can be used by the previous layer to perform it's backward pass.
 * Returns NULL if fails.
 *  
 * @param layer The layer on which the backward pass is performed
 * @param output_gradient The gradient tensor of output of this layer
*/
Tensor* backward_pass(Layer* layer, Tensor* output_gradient);




// void update_weights_biases(Layer* layer, Optimiser* optimiser);  not sure right now on how to use this



#endif
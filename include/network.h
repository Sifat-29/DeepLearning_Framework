#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "loss.h"
#include "optimiser.h"



typedef struct Network {

    Layer* *layers;             // Array of pointers to the layers
    int input_feature_size;     // Number of features of a single sample (Number of cols of sample inputs tensor)
    int n_layers;               // Number of layers in the network
    int capacity;               // For dynamic array resizing

    Loss* loss_func;            // Loss function of the Network
    Optimiser* optimiser;       // The contains it's optimiser

} Network;



// ==========================================
//             Object Management
// ==========================================

/**
 * Creates a new network with the loss function and optimiser as it's internal details.
 * Will later add more parameters to include optimiser hyperparameters
 * 
 * @param input_feature_size Number of features of a single sample
 * @param loss_type Type of the loss function for this network
 * @param opt_type Type of the optimiser (right now only SGD is functional)
 * @param lr Learning rate of the optimiser
*/
Network* create_network(int input_feature_size, loss_function_type loss_type, OptimiserType opt_type, float lr);



/**
 * Completely frees a network, including all it's internal layers.
*/
void free_network(Network** net);



// ==========================================
//             Object Settings
// ==========================================

/**
 * Creates a new layer and adds it to the network.
 * Returns 0 and prints on STDOUT if any error.
 * 
 * @param net Network to which the layer is added.
 * @param n_neurons Number of neurons in this layer.
 * @param prev_n_neurons Number of neurons in the prev layer (the layer connected to this).
 * @param act_func_name Name of the activation functions out of the available ones.
*/
int network_add_layer(Network* net, int n_neurons, activation_function func);



// ==========================================
//                Utilites
// ==========================================

/**
 * Gives new prediction tensor based on the input passed to the network.
 * Returns NULL if any error.
 * 
 * @param net The network which is trained.
 * @param input Input tensor (number_of_inputs x features of single input).
*/
Tensor* network_predict(Network* net, Tensor* input);



/**
 * Trains the network.
 * Returns 0 if any error.
 * 
 * @param net The network which is trained.
 * @param x_train Array of tensor of inputs. Each tensor is (inputs_in_batch x features_of_single_input) 
 * @param y_train Array of tenosr of outputs. Each tensor is (inputs_in_batch x features_of_single_output(also number of neurons in last layer))
 * @param number_of_batches Total number of batches.
 * @param epochs Total number of epochs to train on.
*/
int network_train(Network* net, Tensor* *x_train, Tensor* *y_train, int number_of_batches, int epochs);



#endif
#include "network.h"

#include <stdlib.h>
#include <stdio.h>

#define INITIAL_NETWORK_SIZE        4
#define NETWORK_SIZE_MULTIPLIER     1.5



// ==========================================
//             Object Management
// ==========================================

/**
 * Creates a new network with the loss function and optimiser as it's internal details.
 * Will later add more parameters to include optimiser hyperparameters
 * Returns NUlL if any error
 * 
 * @param input_feature_size Number of features of a single sample
 * @param loss_type Type of the loss function for this network
 * @param opt_type Type of the optimiser (right now only SGD is functional)
 * @param lr Learning rate of the optimiser
*/
Network* create_network(int input_feature_size, loss_function_type loss_type, OptimiserType opt_type, float lr) {
    if (input_feature_size <= 0) {printf("Input_feature_size cannot be zero\n"); return NULL;}

    Network* new_net = (Network*) malloc(sizeof(Network));
    if (!new_net) {printf("Malloc for network failed\n"); return NULL;}

    new_net->input_feature_size = input_feature_size;
    new_net->n_layers = 0;
    new_net->capacity = INITIAL_NETWORK_SIZE;

    new_net->layers = (Layer**) malloc(sizeof(Layer*) * new_net->capacity);
    if (!new_net->layers) {
        printf("Malloc for dynamic array of layers failed\n");
        free(new_net);
        return NULL;
    }

    new_net->loss_func = create_loss(loss_type);
    if (!new_net->loss_func) {
        printf("loss for network could not be created\n");
        free(new_net->layers);
        free(new_net);
        return NULL;
    }

    new_net->optimiser = create_optimiser(opt_type, lr);
    if (!new_net->optimiser) {
        printf("optimiser for network could not be created\n");
        free_loss(&(new_net->loss_func));
        free(new_net->layers);
        free(new_net);
        return NULL;
    }

    return new_net;
}



/**
 * Completely frees a network, including all it's internal layers.
*/
void free_network(Network** net) {
    if (net && *net) {
        for(int i = 0; i < (*net)->n_layers; i++) free_layer(&((*net)->layers[i]));

        free((*net)->layers);

        free_loss(&((*net)->loss_func));

        free_optimiser(&((*net)->optimiser));

        free(*net);
        *net = NULL;
    } 
}



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
int network_add_layer(Network* net, int n_neurons, activation_function func) {
    if (!net) {printf("Network pased is NULL\n"); return 0;}

    int idx_layer = net->n_layers;

    int n_prev_neurons = (idx_layer == 0) ? net->input_feature_size : net->layers[idx_layer - 1]->n_neurons;
    Layer* new_layer = create_layer(n_neurons, n_prev_neurons, func);
    if (!new_layer) {printf("New layer could not be made\n"); return 0;}

    if (idx_layer == net->capacity) {
        int new_capacity = (int)(net->capacity * NETWORK_SIZE_MULTIPLIER);
        Layer* *temp = (Layer**)realloc(net->layers, new_capacity * sizeof(Layer*));

        if (!temp) {
            printf("Realloc failed\n");
            free_layer(&new_layer);
            return 0;
        }

        net->layers = temp;
        net->capacity = new_capacity;
    }

    net->layers[idx_layer] = new_layer;
    net->n_layers++;

    return 1;
}



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
Tensor* network_predict(Network* net, Tensor* input) {
    if (!net || !input) {
        if (!net) printf("The net passed is NULL\n");
        if (!input) printf("The input tensor passed is NULL\n");
        return NULL;
    }

    if (net->n_layers == 0 || input->cols != net->input_feature_size) {
        if (net->n_layers == 0) printf("There are no layers in the neural network\n");
        if (input->cols != net->input_feature_size) printf("Mismatch between features of a single input between network and the input tensor passed\n");
        return NULL;
    }

    Tensor* input_for_current_layer = input;
    Tensor* input_for_next_layer = NULL;

    for (int layer_idx = 0; layer_idx < net->n_layers; layer_idx++) {
        input_for_next_layer = forward_pass(net->layers[layer_idx], input_for_current_layer);
        if (!input_for_next_layer) {printf("Forward pass failed\n"); return NULL;}

        if (layer_idx != 0) free_tensor(&input_for_current_layer);
        input_for_current_layer = input_for_next_layer;
    }

    return input_for_next_layer;
}



/**
 * Trains the network.
 * Return 0 if any error.
 * 
 * @param net The network which is trained.
 * @param x_train Array of tensor of inputs. Each tensor is (inputs_in_batch x features_of_single_input) 
 * @param y_train Array of tenosr of outputs. Each tensor is (inputs_in_batch x features_of_single_output(also number of neurons in last layer))
 * @param number_of_batches Total number of batches.
 * @param epochs Total number of epochs to train on.
*/
int network_train(Network* net, Tensor* *x_train, Tensor* *y_train, int number_of_batches, int epochs) {
    if (!net || !x_train || !y_train || epochs <= 0) {
        if (!net) printf("net given is NULL\n");
        if (!x_train) printf("x_train given is NULL\n");
        if (!y_train) printf("y_train given is NULL\n");
        if (epochs <= 0) printf("Epochs need to be non zero positive integer\n");
        return 0;
    }

    if (net->input_feature_size != x_train[0]->cols) {printf("Mismatch between cols of x_train and network's input feature size\n"); return 0;}

    printf("Start Training... (Batches: %d, Epochs: %d)\n", number_of_batches, epochs);

    int batch_print_interval = number_of_batches / 10;
    if (batch_print_interval == 0) batch_print_interval = 1;

    int epoch_print_interval = epochs / 10;
    if (epoch_print_interval == 0) epoch_print_interval = 1;

    for (int e = 0; e < epochs; e++) {
        float epoch_loss = 0.0f;

        for (int batch_idx = 0; batch_idx < number_of_batches; batch_idx++) {
            if (batch_idx % batch_print_interval == 0) printf("  [Epoch %d] Processing batch %d/%d...\n", e + 1, batch_idx + 1, number_of_batches);

            Tensor* pred = network_predict(net, x_train[batch_idx]);
            if (!pred) {printf("Failed to get a prediction from network\n"); return 0;}

            float current_loss = net->loss_func->loss(pred, y_train[batch_idx]);
            epoch_loss += current_loss;

            Tensor* prev_grad = net->loss_func->derivative(pred, y_train[batch_idx]);
            if (!prev_grad) {printf("Failed to get loss gradient of the prediction\n"); return 0;}

            free_tensor(&pred);

            Tensor* grad = NULL;

            for (int i = net->n_layers - 1; i >= 0; i--) {
                grad = backward_pass(net->layers[i], prev_grad);
                if (!grad) {
                    printf("backward pass failed\n");
                    if (prev_grad) free_tensor(&prev_grad);
                    return 0;
                }

                free_tensor(&prev_grad);
                prev_grad = grad;
            }
            free_tensor(&prev_grad);

            for (int i = 0; i < net->n_layers; i++) optimiser_update(net->optimiser, net->layers[i], i);    /* Can be refactored for security */
        }
        
        if ((e + 1) % epoch_print_interval == 0 || e == 0 || e == epochs - 1) {
            float avg_loss = epoch_loss / number_of_batches;
            printf("Epoch %d/%d | Avg Loss: %.6f\n\n", e + 1, epochs, avg_loss);
        }
    }

    printf("Training Complete.\n");

    return 1;    /* For success */
}

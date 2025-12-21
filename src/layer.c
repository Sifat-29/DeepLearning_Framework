#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>



// ==========================================
//             Object Management
// ==========================================

/**
 * Returns a new layer based on the input parameters.
 * If any error, returns NULL and prints the error.
 * 
 * @param n_neurons Number of neurons in this layer.
 * @param prev_n_neurons Number of neurons in the prev layer (the layer connected to this).
 * @param act_func_name Name of the activation functions out of the available ones.
*/
Layer* create_layer(int n_neurons, int prev_n_neurons, activation_function act_func_name) {
    if (n_neurons <= 0 || prev_n_neurons <= 0) {
        if (n_neurons <= 0) printf("n_neurons entered is invalid (<= 0)\n");
        if (prev_n_neurons <= 0) printf("prev_n_neurons entered is invalid (<= 0)\n");
        return NULL;
    }

    Layer* new_layer = (Layer*) malloc(sizeof(Layer));
    if (!new_layer) {printf("Malloc for new layer failed\n"); return NULL;}

    new_layer->n_neurons = n_neurons;
    new_layer->n_neurons_prev = prev_n_neurons;

    float limit = sqrt(6.0f / (float)prev_n_neurons);    /* Formula to understand later */
    new_layer->weights = create_tensor_random(prev_n_neurons, n_neurons, -limit, limit);
    if (!new_layer->weights) {
        printf("Error in creating tensor for weights\n"); 
        free(new_layer);
        return NULL;
    }

    new_layer->biases = create_tensor_value(1, n_neurons, 0.01f); 
    if (!new_layer->biases) {
        printf("Error in creating tensor for biases\n"); 
        free_tensor(&(new_layer->weights));
        free(new_layer);
        return NULL;
    }
    
    new_layer->d_weights = NULL;
    new_layer->d_biases = NULL;
    new_layer->input_transpose_cache = NULL;
    new_layer->z_cache = NULL;

    new_layer->activation = create_activation(act_func_name);
    if (!new_layer->activation) {
        printf("Error in creating activation for the layer\n"); 
        free_tensor(&(new_layer->biases));
        free_tensor(&(new_layer->weights));
        free(new_layer);
        return NULL;
    }

    return new_layer;
}



/**
 * Completely frees the layer
 * 
*/
void free_layer(Layer** layer) {
    if (layer && *layer) {
        if ((*layer)->weights) free_tensor(&((*layer)->weights));
        if ((*layer)->biases) free_tensor(&((*layer)->biases));

        if ((*layer)->d_weights) free_tensor(&((*layer)->d_weights));
        if ((*layer)->d_biases) free_tensor(&((*layer)->d_biases));

        if ((*layer)->z_cache) free_tensor(&((*layer)->z_cache));
        if ((*layer)->input_transpose_cache) free_tensor(&((*layer)->input_transpose_cache));

        if ((*layer)->activation) free_activation(&((*layer)->activation));

        free(*layer);
        *layer = NULL; 
    }
}



// ==========================================
//          Training and Prediction
// ==========================================

/**
 * Returns the output of the forward pass performed on the layer with a given input.
 * Returns NULL if fails.
 * 
 * Z = X @ W + B
 * 
 * @param layer The layer on which the forward pass is performed
 * @param input The input tensor (batch_size x n_neurons_prev) on which the forward pass is performed
*/
Tensor* forward_pass(Layer* layer, Tensor* input) {
    if (!layer || !input) {
        if (!layer) printf("Layer is NULL\n");
        if (!input) printf("Input tensor is NULL\n");
        return NULL;
    }

    if (layer->input_transpose_cache) free_tensor(&(layer->input_transpose_cache));
    Tensor* input_transpose = tensor_transpose(input);
    if (!input_transpose) {printf("Transpose of input failed \n"); return NULL;}
    layer->input_transpose_cache = input_transpose;
    

    Tensor* z = tensor_multiplication(input, layer->weights);
    if (!z) {printf("Matrix multiplication failed\n"); return NULL;}
    tensor_row_addition_inplace(z, layer->biases);

    if (layer->z_cache) free_tensor(&(layer->z_cache));
    layer->z_cache = z;

    Tensor* res = tensor_deepcopy(z);
    if (!res) {printf("Tensor deepcopy failed on a has failed\n"); return NULL;}

    layer->activation->forward_inplace(res);    /* Apply activation function to the res tensor in place */
    
    return res;
}



/**
 * Returns the gradient of output this layer so it can be used by the previous layer to perform it's backward pass.
 *  
 * @param layer The layer on which the backward pass is performed
 * @param output_gradient The gradient tensor of output of this layer
*/
Tensor* backward_pass(Layer* layer, Tensor* output_gradient) {
    if (!layer || !output_gradient) {
        if (!layer) printf("Layer is NULL\n");
        if (!output_gradient) printf("output_gradient tensor is NULL\n");
        return NULL;
    }
    
    if (!layer->z_cache) {printf("a_cache is NULL\n"); return NULL;}
    Tensor* a_prime_z = layer->activation->backward(layer->z_cache);
    if (!a_prime_z) {printf("a_prime_z could not be computed\n"); return NULL;}
    Tensor* dz = tensor_multiplication_hadamard(output_gradient, a_prime_z); 
    if (!dz) {printf("dz could not be computed\n"); return NULL;}


    if (!layer->input_transpose_cache) {printf("input_transpose_cache is NULL\n"); return NULL;}
    if (layer->d_weights) free_tensor(&(layer->d_weights));
    layer->d_weights = tensor_multiplication(layer->input_transpose_cache, dz);

    if (layer->d_biases) free_tensor(&(layer->d_biases));
    layer->d_biases = tensor_add_cols(dz);


    Tensor* wt = tensor_transpose(layer->weights);
    if (!wt) {printf("wt could not be computed\n"); return NULL;}

    Tensor* dx = tensor_multiplication(dz, wt);
    if (!dx) {printf("dx could not be computed\n"); return NULL;}

    free_tensor(&a_prime_z);
    free_tensor(&dz);
    free_tensor(&wt);

    return dx;
}

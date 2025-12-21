#include "optimiser.h"

#include <stdlib.h>
#include <stdio.h>



// ==========================================
//             Internal Helpers
// ==========================================

void _sgd_update(Optimiser* opt, Layer* layer);
void _sgd_m_update(Optimiser* opt, Layer* layer, int layer_idx);
void _adam_update(Optimiser* opt, Layer* layer, int layer_idx);



// ==========================================
//             Object Management
// ==========================================

/**
 * Creates the optimizer object with the only the learning rate, will later add methods to set other hyperparameters for SGD+M and Adam.
 * Returns NULL if any error.
 * 
 * @param type Type of the optimiser out of the available in the enum.
 * @param lr Learning rate
 */
Optimiser* create_optimiser(OptimiserType type, float lr) {
    if (lr < 0.0) {printf("Learning rate cannot be negative\n"); return NULL;}

    Optimiser* new_opt = (Optimiser*) malloc(sizeof(Optimiser));
    if (!new_opt) {printf("malloc for optimiser failed\n"); return NULL;}

    new_opt->type = type;
    new_opt->learning_rate = lr;
    
    new_opt->beta1 = 0.0;       /* Will later add default value here */
    new_opt->beta2 = 0.0;       /* Will later add default value here */
    new_opt->epsilon = 0.0;     /* Will later add default value here */

    new_opt->time_step = 0;

    return new_opt;
}



/**
 * Completely frees the optimiser.
 * Will need to change later for caches
 */
void free_optimiser(Optimiser** opt) {
    if (opt && *opt) {
        free(*opt);
        *opt = NULL;
    }
}



// ==========================================
//             Update Logic
// ==========================================

/**
 * Performs the weight update on a SINGLE layer.
 * 
 * @param opt The optimizer
 * @param layer The layer to update
 * @param layer_idx Index of the layer in the network, used by SGD+M and Adam
 */
void optimiser_update(Optimiser* opt, Layer* layer, int layer_index) {
    switch (opt->type)
    {
    case SGD:
        _sgd_update(opt, layer);
        break;

    case SGD_MOMENTUM:
        _sgd_m_update(opt, layer, layer_index);
        break;
    
    case ADAM:
        _adam_update(opt, layer, layer_index);
        break;
        
    default:
        _sgd_update(opt, layer);
        break;
    }
}



// ==========================================
//         Update Specific to Type
// ==========================================


void _sgd_update(Optimiser* opt, Layer* layer) {
    tensor_add_scaled_inplace(layer->weights, layer->d_weights, -opt->learning_rate);
    tensor_add_scaled_inplace(layer->biases, layer->d_biases, -opt->learning_rate);
}



void _sgd_m_update(Optimiser* opt, Layer* layer, int layer_idx) {
    (void)opt;
    (void)layer;
    (void)layer_idx;
}



void _adam_update(Optimiser* opt, Layer* layer, int layer_idx) {
    (void)opt;
    (void)layer;
    (void)layer_idx;
}
#ifndef OPTIMISER_H
#define OPTIMISER_H

#include "tensor.h"
#include "layer.h"



typedef enum { 
    SGD,                    
    SGD_MOMENTUM,           // Later
    ADAM                    // Later
} OptimiserType;



typedef struct Optimiser {
    
    OptimiserType type;         // Type of the optimiser available in the enum
    float learning_rate;        // Learning rate of the optimiser

    /* Parameters to be later used by SGD+M and Adam */
    float beta1;                // Used by SGD+M and Adam
    float beta2;                // Used by Adam
    float epsilon;              // Used by Adam
    int time_step;              // Used by Adam

    /* Will add internal caches for SGD+M and Adam later */

} Optimiser;




// ==========================================
//             Object Management
// ==========================================

/**
 * Creates the optimizer object with the only the learning rate, will later add methods to set other hyperparameters for SGD+M and Adam.
 * 
 * @param type Type of the optimiser out of the available in the enum.
 * @param lr Learning rate
 */
Optimiser* create_optimiser(OptimiserType type, float lr);



/**
 * Completely frees the optimiser.
 */
void free_optimiser(Optimiser** opt);



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
void optimiser_update(Optimiser* opt, Layer* layer, int layer_index);



#endif
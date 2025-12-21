#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"



/* Enum containing all the activation functions */
typedef enum { MSE, CATEGORICAL_CROSSENTROPY } loss_function_type;



typedef struct Loss {
    
    float (*loss)(Tensor* pred, Tensor* target);                // Calculates loss for the given prediction and target
    Tensor* (*derivative)(Tensor* pred, Tensor* target);        // Calculates gradient wrt prediction
    loss_function_type type;                                    // type of the loss function

} Loss;



// ==========================================
//             Object Management
// ==========================================

/**
 * Returns a new activation object (contains the forward and backward functions).
 * 
 * @param func the activation function which is to be used.
*/
Loss* create_loss(loss_function_type func);



/**
 * Completely frees the loss object
*/
void free_loss(Loss** loss);



#endif
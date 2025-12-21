#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"



/* Enum containing all the activation functions */
typedef enum { RELU, SIGMOID, SOFTMAX, LINEAR } activation_function;



typedef struct Activation {

    void (*forward_inplace)(Tensor*);           // Forward fuction
    Tensor* (*backward)(Tensor*);       // The derivative function
    activation_function func;           // For debugging?
    
} Activation;



// ==========================================
//             Object Management
// ==========================================

/**
 * Returns a new activation object (contains the forward and backward functions).
 * 
 * @param func the activation function which is to be used.
*/
Activation* create_activation(activation_function func);



/**
 * Completely frees the activation object
*/
void free_activation(Activation** act);



#endif
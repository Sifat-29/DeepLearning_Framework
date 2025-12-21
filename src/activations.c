#include "activations.h"

#include <stdlib.h>
#include <stdio.h>



// ==========================================
//             Internal Helpers
// ==========================================

void _relu_inplace(Tensor* t);
float _apply_relu_to_element(float x);
Tensor* _d_relu(Tensor* t);
float _apply_d_relu_to_element(float x);

/* Will implement later */ 
void _sigmoid_inplace(Tensor* t);
Tensor* _d_sigmoid(Tensor* t);
void _softmax_inplace(Tensor* t);
Tensor* _d_softmax(Tensor* t);


void _linear_inplace(Tensor* t);
float _apply_d_linear_element(float x);
Tensor* _d_linear(Tensor* t);



// ==========================================
//             Object Management
// ==========================================

/**
 * Returns a new activation object (contains the forward and backward functions).
 * 
 * @param func the activation function which is to be used.
*/
Activation* create_activation(activation_function func) {
    Activation* new_activation = (Activation*) malloc(sizeof(Activation));
    if (!new_activation) {printf("Malloc failed for activation\n"); return NULL;}

    new_activation->func = func;

    switch (func)
    {
    case RELU:
        new_activation->forward_inplace = _relu_inplace;
        new_activation->backward = _d_relu;
        break;

    // case SIGMOID:
    //     new_activation->forward_inplace = _sigmoid_inplace;
    //     new_activation->backward = _d_sigmoid;
    //     break;
        
    // case SOFTMAX:
    //     new_activation->forward_inplace = _softmax_inplace;
    //     new_activation->backward = _d_softmax;
    //     break;

    case LINEAR:
        new_activation->forward_inplace = _linear_inplace;
        new_activation->backward = _d_linear;
        break;

    default:    /* RELU is default */
        printf("Unknown activation type, defaulting to ReLU\n");
        new_activation->forward_inplace = _relu_inplace;
        new_activation->backward = _d_relu;
        break;
    }

    return new_activation;
}



/**
 * Completely frees the activation object
*/
void free_activation(Activation** act) {
    if (act && *act) {
        free(*act);
        *act = NULL;
    }
}



// ===================================
//              Relu
// ===================================

void _relu_inplace(Tensor* t) {
    if (!t) {printf("Tensor received is NULL\n"); return;}
    tensor_apply_func_inplace(t, _apply_relu_to_element);
}


float _apply_relu_to_element(float x) {
    return (x > 0.0f) ? x : 0.01f * x;
}

float _apply_d_relu_to_element(float x) {
    return (x > 0.0f) ? 1.0f : 0.01f;
}

Tensor* _d_relu(Tensor* t) {
    if (!t) {printf("Tensor received is NULL\n"); return NULL;}

    Tensor* res = tensor_deepcopy(t);
    if (!res) {printf("Tensor deepcopy failed\n"); return NULL;}

    tensor_apply_func_inplace(res, _apply_d_relu_to_element);
    return res;
}



// ===================================
//              Sigmoid
// ===================================

void _sigmoid_inplace(Tensor* t) {
    printf("Sigmoid Forward not implemented yet.\n");
}
Tensor* _d_sigmoid(Tensor* t) {
    printf("Sigmoid Backward not implemented yet.\n");
    return NULL;
}



// ===================================
//              Softmax
// ===================================

void _softmax_inplace(Tensor* t) {
    printf("Softmax Forward not implemented yet.\n");
}
Tensor* _d_softmax(Tensor* t) {
    printf("Softmax Backward not implemented yet.\n");
    return NULL;
}



// ===================================
//              Linear
// ===================================

void _linear_inplace(Tensor* t) {
    return;
}

float _apply_d_linear_element(float x) {
    return 1.0f;
}

Tensor* _d_linear(Tensor* t) {
    if (!t) return NULL;
    
    Tensor* res = tensor_deepcopy(t);
    if (!res) return NULL;

    tensor_apply_func_inplace(res, _apply_d_linear_element);
    return res;
}
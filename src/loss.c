#include "loss.h"

#include <stdlib.h>
#include <stdio.h>


// ==========================================
//             Internal Helpers
// ==========================================

float _mse_loss(Tensor* pred, Tensor* target);
Tensor* _mse_derivative(Tensor* pred, Tensor* target);



// ==========================================
//             Object Management
// ==========================================

/**
 * Returns a new activation object (contains the forward and backward functions).
 * Returns NULL if error.
 * 
 * @param func the activation function which is to be used.
*/
Loss* create_loss(loss_function_type func) {
    Loss* new_loss = (Loss*) malloc(sizeof(Loss));
    if (!new_loss) {printf("malloc for loss function failed"); return NULL;}

    new_loss->type = func;

    switch (func)
    {
    case MSE:
        new_loss->loss = _mse_loss;
        new_loss->derivative = _mse_derivative;
        break;

    // case CATEGORICAL_CROSSENTROPY:    Implement later
    //     break;
    
    default:
        new_loss->loss = _mse_loss;
        new_loss->derivative = _mse_derivative;
        break;
    }

    return new_loss;
}



/**
 * Completely frees the loss object
*/
void free_loss(Loss** loss) {
    if (loss && *loss) {
        free(*loss);
        *loss = NULL;
    }
}



// ==========================================
//            Mean Squared Error 
// ==========================================

/**
 * Returns MSE loss on basis of target and prediction tensor.
 * Prints to STDOUT in case of any error (and return 0).
 * 
 * @param pred The tensor predidcted by the model
 * @param target What the data indicates
*/
float _mse_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) {
        if (!pred) printf("pred is NULL"); 
        if (!target) printf("target is NULL");
        return 0.0f;
    }

    if (pred->cols != target->cols || pred->rows != target->rows) {
        if (pred->cols != target->cols) printf("Mismatch between cols of pred and target\n");
        if (pred->rows != target->rows) printf("Mismatch between rows of pred and target\n");
        return 0.0f;
    }

    float error = 0.0f;
    
    for (int input = 0; input < pred->rows; input++) for (int output_feature = 0; output_feature < pred->cols; output_feature++) {
        error += (target->data[input*target->cols + output_feature] - pred->data[input*pred->cols + output_feature])*(target->data[input*target->cols + output_feature] - pred->data[input*pred->cols + output_feature]);
    }

    return error / (float)(pred->cols * pred->rows);
}



/**
 * Return gradient tensor of the mse loss wrt output of the last layer.
 * Returns NULL in case of any error.
 * 
 * @param pred The tensor predidcted by the model
 * @param target What the data indicates
*/
Tensor* _mse_derivative(Tensor* pred, Tensor* target) {
    if (!pred || !target) {
        if (!pred) printf("pred is NULL"); 
        if (!target) printf("target is NULL");
        return NULL;
    }

    if (pred->cols != target->cols || pred->rows != target->rows) {
        if (pred->cols != target->cols) printf("Mismatch between cols of pred and target\n");
        if (pred->rows != target->rows) printf("Mismatch between rows of pred and target\n");
        return NULL;
    }

    Tensor* res = tensor_deepcopy(pred);
    if (!res) {printf("Tensor deepcopy failed in _mse_derivative"); return NULL;}

    float factor = 2.0f / (float)(pred->cols * pred->rows);
    
    for (int input = 0; input < pred->rows; input++) for (int output_feature = 0; output_feature < pred->cols; output_feature++) {
        res->data[input*pred->cols + output_feature] = factor*(pred->data[input*pred->cols + output_feature] - target->data[input*target->cols + output_feature]);
    }

    return res;
}
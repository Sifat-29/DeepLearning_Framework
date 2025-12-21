#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>



// ==========================================
//             Internal Helpers
// ==========================================

Tensor* _create_tensor(int rows, int cols);
float _random_float_range(float min, float max);



// ==========================================
//             Object Management
// ==========================================

/**
 * Initialises the API by seeding for the random API calls
*/
void init_tensor_api() {
    srand(time(NULL));
}



/**
 * Allocates and returns a tensor with uninitialised (garbage) values.
 * Values are then set by the functions calling this internal method.
 * Returns NULL if any error and prints the cause on STDOUT.
 * 
 * @param rows number of rows of tensor
 * @param cols number of cols of tensor 
 */
Tensor* _create_tensor(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        if (rows <= 0) printf("Number of rows received is less than 1\n");
        if (cols <= 0) printf("Number of cols received is less than 1\n");
        return NULL;
    }

    Tensor* tensor_created = (Tensor *) malloc(sizeof(Tensor));
    if (!tensor_created) {
        printf("Malloc failed for creating a tensor\n"); 
        return NULL;
    }

    tensor_created->rows = rows;
    tensor_created->cols = cols;

    tensor_created->data = (float *) malloc(rows * cols * sizeof(float));
    if (!tensor_created->data) {
        printf("Malloc failed for creating internals of tensor\n"); 
        free(tensor_created);
        return NULL;
    }

    return tensor_created;
} 



/**
 * Returns pointer to a tensor of (rows x cols) with the values initialised to the value given.
 * Returns NULL if any error.
 * 
 * @param rows number of rows of tensor
 * @param cols number of cols of tensor 
 * @param value the float value to which the tensor is initialised
 */
Tensor* create_tensor_value(int rows, int cols, float value) {
    Tensor* uninit_tensor = _create_tensor(rows, cols);
    if (!uninit_tensor) return NULL;
    
    for (int i = 0; i < uninit_tensor->rows; i++) for (int j = 0; j < uninit_tensor->cols; j++) uninit_tensor->data[i*cols + j] = value; 

    return uninit_tensor;
}



/**
 * Returns pointer to a tensor of (rows x cols) with the values initialised to a random number between min and max.
 * Returns NULL if any error.
 * 
 * @param rows number of rows of tensor
 * @param cols number of cols of tensor 
 * @param min minimum random value (inclusive)
 * @param max maximum random value (inclusive)
 */
Tensor* create_tensor_random(int rows, int cols, float min, float max) {
    if (min > max) {
        printf("Min is greater than Max\n");
        return NULL;
    }

    Tensor* uninit_tensor = _create_tensor(rows, cols);
    if (!uninit_tensor) return NULL;
    
    for (int i = 0; i < uninit_tensor->rows; i++) for (int j = 0; j < uninit_tensor->cols; j++) uninit_tensor->data[i*cols + j] = _random_float_range(min, max); 

    return uninit_tensor;
}



/**
 * Returns a random float in [min, max].
 * 
 * @param min minimum random value (inclusive)
 * @param max maximum random value (inclusive)
 */
float _random_float_range(float min, float max) {
    float scale =  (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}



/**
 * Creates and returns deepcopy of the tensor input.
 * 
 * @param tensor deepcopy of this tensor is returned.
 */
Tensor* tensor_deepcopy(const Tensor* tensor) {
    if (!tensor) {printf("Tensor given is NULL\n"); return NULL;}

    Tensor* new_tensor = _create_tensor(tensor->rows, tensor->cols);
    if (!new_tensor) {printf("Unable to create new tensor\n"); return NULL;}

    for (int i = 0; i < new_tensor->rows; i++) for (int j = 0; j < new_tensor->cols; j++) new_tensor->data[i*tensor->cols + j] = tensor->data[i*tensor->cols + j];

    return new_tensor;
}



/**
 * Frees the tensor pointer and sets it to NULL
 */
void free_tensor(Tensor** tensor) {
    if (tensor && *tensor) {
        Tensor* t = *tensor;
        if (t->data) free(t->data);

        free(t);
        *tensor = NULL; 
    }
}



// ==========================================
//             Operations (new object)
// ==========================================

/**
 * Returns a new Tensor which is the result of addition of t1 and t2 (t1 + t2).
 * Returns NULL if the number of rows and cols do not match.
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_addition(const Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return NULL;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return NULL;
    }

    Tensor* t_new = _create_tensor(t1->rows, t1->cols);

    for (int i = 0; i < t_new->rows; i++) for (int j = 0; j < t_new->cols; j++) t_new->data[i*t_new->cols + j] = t1->data[i*t1->cols + j] + t2->data[i*t2->cols + j];

    return t_new;
}



/**
 * Returns a new Tensor which is the result of subtraction of t2 from t1 (t1 - t2).
 * Returns NULL if the number of rows and cols do not match.
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_subtraction(const Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return NULL;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return NULL;
    }

    Tensor* t_new = _create_tensor(t1->rows, t1->cols);

    for (int i = 0; i < t_new->rows; i++) for (int j = 0; j < t_new->cols; j++) t_new->data[i*t_new->cols + j] = t1->data[i*t1->cols + j] - t2->data[i*t2->cols + j];

    return t_new;
}



/**
 * Returns a new Tensor (t1->rows x t2->cols) which is the result of matrix multiplication of t1 and t2 (t1 @ t2).
 * Returns NULL if the number of cols of t1 and rows of t2 do not match.
 * Simple Matrix multiplication (not in use now)
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_multiplication_v1(const Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return NULL;
    }

    if (t1->cols != t2->rows) {
        if (t1->cols != t2->rows) printf("The number of cols of t1 and rows of t2 do not match\n");
        return NULL;
    }

    Tensor* t_new = create_tensor_value(t1->rows, t2->cols, 0.0);

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t2->cols; j++) for (int k = 0; k < t1->cols; k++) {
        t_new->data[i*t_new->cols + j] += t1->data[i*t1->cols + k] * t2->data[k*t2->cols + j];
    }  

    return t_new;
}



/**
 * Returns a new Tensor (t1->rows x t2->cols) which is the result of matrix multiplication of t1 and t2 (t1 @ t2).
 * Returns NULL if the number of cols of t1 and rows of t2 do not match.
 * Optimised (still O(n^3) but much better caching, reduced time by 20%!!!)
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_multiplication(const Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return NULL;
    }

    if (t1->cols != t2->rows) {
        if (t1->cols != t2->rows) printf("The number of cols of t1 and rows of t2 do not match\n");
        return NULL;
    }

    Tensor* result = create_tensor_value(t1->rows, t2->cols, 0.0f);
    
    // OPTIMISATION: Using transposed copy of t2
    // This is to traverse both t1 and t2_t in row-major order (sequentially).
    Tensor* t2_t = tensor_transpose(t2); 

    for (int i = 0; i < t1->rows; i++) {
        for (int j = 0; j < t2->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < t1->cols; k++) {
                sum += t1->data[i * t1->cols + k] * t2_t->data[j * t2_t->cols + k];
            }
            result->data[i * result->cols + j] = sum;
        }
    }

    free_tensor(&t2_t);
    return result;
}



/**
 * Returns a new Tensor which is the result of matrix hadamard multiplication of t1 and t2 (element wise multiplication).
 * Returns NULL if the number of rows and cols do not match.
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_multiplication_hadamard(const Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return NULL;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return NULL;
    }

    Tensor* t_new = _create_tensor(t1->rows, t1->cols);

    for (int i = 0; i < t_new->rows; i++) for (int j = 0; j < t_new->cols; j++) t_new->data[i*t_new->cols + j] = t1->data[i*t1->cols + j] * t2->data[i*t2->cols + j];

    return t_new;
}



/**
 * Returns a new Tensor which is the transpose of the tensor.
 * Returns NULL if the tensor is NULL
 * 
 * @param t1 the tensor
 */
Tensor* tensor_transpose(const Tensor* tensor) {
    if (!tensor) {
        if (!tensor) printf("tensor is NULL\n");
        return NULL;
    }

    Tensor* t_new = _create_tensor(tensor->cols, tensor->rows);

    for (int i = 0; i < t_new->rows; i++) for (int j = 0; j < t_new->cols; j++) {
        t_new->data[i*t_new->cols + j] = tensor->data[j*tensor->cols + i];
    }

    return t_new;
}



/**
 * Returns a new Tensor which is the result of collapsing all rows (by summing) (rows x cols ---> 1 x cols).
 * Returns NULL if the tensor is NULL
 * 
 * @param t1 the tensor
 */
Tensor* tensor_add_cols(const Tensor* tensor) {
    if (!tensor) {
        if (!tensor) printf("tensor is NULL\n");
        return NULL;
    }

    Tensor* t_new = _create_tensor(1, tensor->cols);

    for (int i = 0; i < tensor->cols; i++) {
        float sum = 0.0;
        for (int j = 0; j < tensor->rows; j++) {
            sum += tensor->data[j*tensor->cols + i];
        }
        t_new->data[i] = sum;
    }

    return t_new;
}



// ==========================================
//      Operations (in-place, modify t1)
// ==========================================

/**
 * t1 = t1 + t2
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
void tensor_addition_inplace(Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return;
    }

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t1->cols; j++) t1->data[i*t1->cols + j] = t1->data[i*t1->cols + j] + t2->data[i*t2->cols + j];
}



/**
 * t1 = t1 - t2
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
void tensor_subtraction_inplace(Tensor* t1, const Tensor* t2){
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return ;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return;
    }

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t1->cols; j++) t1->data[i*t1->cols + j] = t1->data[i*t1->cols + j] - t2->data[i*t2->cols + j];
}



/**
 * t1[i][j] = t1[i][j] * t2[i][j]
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
void tensor_multiplication_hadamard_inplace(Tensor* t1, const Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return;
    }

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t1->cols; j++) t1->data[i*t1->cols + j] = t1->data[i*t1->cols + j] * t2->data[i*t2->cols + j];
}



/**
 * t1 = t1 + (t2 * scalar) 
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 * @param scaler the scaler
 */
void tensor_add_scaled_inplace(Tensor* t1, const Tensor* t2, float scaler) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return;
    }

    if (t1->rows != t2->rows || t1->cols != t2->cols) {
        if (t1->rows != t2->rows) printf("The number of rows do not match\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return;
    }

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t1->cols; j++) t1->data[i*t1->cols + j] = t1->data[i*t1->cols + j] + scaler * t2->data[i*t2->cols + j];
}



/**
 * t = t * scalar
 * 
 * @param t1 the first tensor
 * @param scaler the scaler
 */
void tensor_scale_inplace(Tensor* t, float scaler) {
    if (!t) {
        if (!t) printf("t1 is NULL\n");
        return;
    }

    for (int i = 0; i < t->rows; i++) for (int j = 0; j < t->cols; j++) t->data[i*t->cols + j] = t->data[i*t->cols + j] * scaler;
}



/**
 * Adds t2 to each row of t1.
 * 
 * @param t1 the first tensor  (rows x cols) 
 * @param t2 the second tensor (1 x cols)
 */
void tensor_row_addition_inplace(Tensor* t1, Tensor* t2) {
    if (!t1 || !t2) {
        if (!t1) printf("t1 is NULL\n");
        if (!t2) printf("t2 is NULL\n");
        return;
    }

    if (t2->rows != 1 || t1->cols != t2->cols) {
        if (t2->rows != 1) printf("The number of rows of t2 is not 1\n");
        if (t1->cols != t2->cols) printf("The number of cols do not match\n");
        return;
    }

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t1->cols; j++) t1->data[i*t1->cols + j] = t1->data[i*t1->cols + j] + t2->data[j];
}



/**
 * t[i][j] = func(t[i][j])
 * 
 * @param t1 the first tensor
 * @param scaler the scaler
 */
void tensor_apply_func_inplace(Tensor* t1, float (*func)(float)) {
        if (!t1 || !func) {
        if (!t1) printf("t1 is NULL\n");
        if (!func) printf("func is NULL\n");
        return;
    }

    for (int i = 0; i < t1->rows; i++) for (int j = 0; j < t1->cols; j++) t1->data[i*t1->cols + j] = func(t1->data[i*t1->cols + j]);
}



// ==========================================
//             Object Viewing
// ==========================================

/**
 * Pretty prints the tensor in a standard matrix format.
 */
void print_tensor(const Tensor* tensor) {
    if (!tensor) {
        printf("NULL pointer received for printing\n");
        return;
    }

    printf("Tensor (Rows=%d, Cols=%d):\n", tensor->rows, tensor->cols);

    for (int i = 0; i < tensor->rows; i++) {
        printf("[");

        for (int j = 0; j < tensor->cols; j++) {
            printf("  %8.4f", tensor->data[i*tensor->cols + j]);
        }

        printf(" ]\n");
    }
}

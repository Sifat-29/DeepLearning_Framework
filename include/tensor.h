#ifndef TENSOR_H
#define TENSOR_H



typedef struct Tensor {
    float *data;           // Matrix of floats
    int rows;              // Rows of matrix 
    int cols;              // Columns of matrix 
} Tensor;



// ==========================================
//             Object Management
// ==========================================

/**
 * Initialises the API by seeding for the random API calls
*/
void init_tensor_api();



/**
 * Returns pointer to a tensor of (rows x cols) with the values initialised to the value given.
 * Returns NULL if any error.
 * 
 * @param rows number of rows of tensor
 * @param cols number of cols of tensor 
 * @param value the float value to which the tensor is initialised
 */
Tensor* create_tensor_value(int rows, int cols, float value);



/**
 * Returns pointer to a tensor of (rows x cols) with the values initialised to a random number between min and max.
 * Returns NULL if any error.
 * 
 * @param rows number of rows of tensor
 * @param cols number of cols of tensor 
 * @param min minimum random value (inclusive)
 * @param max maximum random value (inclusive)
 */
Tensor* create_tensor_random(int rows, int cols, float min, float max);



/**
 * Creates and returns deepcopy of the tensor input.
 * 
 * @param tensor deepcopy of this tensor is returned.
 */
Tensor* tensor_deepcopy(const Tensor* tensor);



/**
 * Frees the tensor pointer and sets it to NULL
 * 
 */
void free_tensor(Tensor** tensor);



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
Tensor* tensor_addition(const Tensor* t1, const Tensor* t2);



/**
 * Returns a new Tensor which is the result of subtraction of t2 from t1 (t1 - t2).
 * Returns NULL if the number of rows and cols do not match.
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_subtraction(const Tensor* t1, const Tensor* t2);



/**
 * Returns a new Tensor (t1->rows x t2->cols) which is the result of matrix multiplication of t1 and t2 (t1 @ t2).
 * Returns NULL if the number of cols of t1 and rows of t2 do not match.
 * Simple Matrix multiplication (will optimise later)
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_multiplication(const Tensor* t1, const Tensor* t2);



/**
 * Returns a new Tensor which is the result of matrix hadamard multiplication of t1 and t2 (element wise multiplication).
 * Returns NULL if the number of rows and cols do not match.
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
Tensor* tensor_multiplication_hadamard(const Tensor* t1, const Tensor* t2);



/**
 * Returns a new Tensor which is the transpose of the tensor.
 * Returns NULL if the tensor is NULL
 * 
 * @param t1 the tensor
 */
Tensor* tensor_transpose(const Tensor* tensor);



/**
 * Returns a new Tensor which is the result of collapsing all rows (by summing) (rows x cols ---> 1 x cols).
 * Returns NULL if the tensor is NULL
 * 
 * @param t1 the tensor
 */
Tensor* tensor_add_cols(const Tensor* tensor);



// ==========================================
//      Operations (in-place, modify t1)
// ==========================================

/**
 * t1 = t1 + t2
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
void tensor_addition_inplace(Tensor* t1, const Tensor* t2);



/**
 * t1 = t1 - t2
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
void tensor_subtraction_inplace(Tensor* t1, const Tensor* t2);



/**
 * t1[i][j] = t1[i][j] * t2[i][j]
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 */
void tensor_multiplication_hadamard_inplace(Tensor* t1, const Tensor* t2);



/**
 * t1 = t1 + (t2 * scalar) 
 * 
 * @param t1 the first tensor
 * @param t2 the second tensor
 * @param scaler the scaler
 */
void tensor_add_scaled_inplace(Tensor* t1, const Tensor* t2, float scaler);



/**
 * t = t * scalar
 * 
 * @param t1 the first tensor
 * @param scaler the scaler
 */
void tensor_scale_inplace(Tensor* t, float scaler);



/**
 * Adds t2 to each row of t1.
 * 
 * @param t1 the first tensor  (rows x cols) 
 * @param t2 the second tensor (1 x cols)
 */
void tensor_row_addition_inplace(Tensor* t1, Tensor* t2);



/**
 * t[i][j] = func(t[i][j])
 * 
 * @param t1 the first tensor
 * @param scaler the scaler
 */
void tensor_apply_func_inplace(Tensor* t1, float (*func)(float));



// ==========================================
//             Object Viewing
// ==========================================

/**
 * Pretty prints the tensor in a standard matrix format.
 */
void print_tensor(const Tensor* tensor);



#endif

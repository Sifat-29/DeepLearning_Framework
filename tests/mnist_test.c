#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "network.h"
#include "tensor.h"



// ==========================================
//             Configuration
// ==========================================
#define BATCH_SIZE 64
#define EPOCHS 10
#define LEARNING_RATE 0.1f



// ==========================================
//             Helper Prototypes
// ==========================================
Network* get_network(int n_features);
int load_mnist_csv(const char* filename, Tensor*** x_data, Tensor*** y_data, int* num_samples, int* n_features);
void create_mini_batches(Tensor** x_in, Tensor** y_in, int total_samples, int batch_size, Tensor*** x_out, Tensor*** y_out, int* total_batches);
void free_mnist_data(Tensor** x_data, Tensor** y_data, int count);
int get_predicted_class(Tensor* pred);



// ==========================================
//                 Main
// ==========================================

int main() {
    init_tensor_api();

    Tensor** x_raw = NULL;
    Tensor** y_raw = NULL;
    int raw_count = 0;
    int n_features = 0;

    printf("\n[1/6] Loading Raw Training Data...\n");
    if (!load_mnist_csv("datasets/MNIST/mnist_train.csv", &x_raw, &y_raw, &raw_count, &n_features)) {
        return 1;
    }
    printf("Loaded %d raw samples.\n", raw_count);

    printf("\n[2/6] Creating Mini-Batches (Batch Size: %d)...\n", BATCH_SIZE);
    
    Tensor** x_batched = NULL;
    Tensor** y_batched = NULL;
    int n_batches = 0;

    create_mini_batches(x_raw, y_raw, raw_count, BATCH_SIZE, &x_batched, &y_batched, &n_batches);

    free_mnist_data(x_raw, y_raw, raw_count);
    printf("Created %d batches. Raw data freed.\n", n_batches);

    printf("\n[3/6] Building Network\n");
    
    Network* net = get_network(n_features); 

    
    printf("\n[4/6] Training for %d Epochs...\n", EPOCHS);
    
    network_train(net, x_batched, y_batched, n_batches, EPOCHS);


    free_mnist_data(x_batched, y_batched, n_batches);

    
    printf("\n[5/6] Loading Test Data...\n");
    
    Tensor** x_test = NULL;
    Tensor** y_test = NULL;
    int test_samples = 0;
    int f_test = 0;

    if (!load_mnist_csv("datasets/MNIST/mnist_test.csv", &x_test, &y_test, &test_samples, &f_test)) {
        free_network(&net);
        return 1;
    }

   

    printf("\n[6/6] Evaluating Accuracy on %d samples...\n", test_samples);
    
    int correct = 0;
    for (int i = 0; i < test_samples; i++) {
        Tensor* pred = network_predict(net, x_test[i]);
        
        if (get_predicted_class(pred) == get_predicted_class(y_test[i])) {
            correct++;
        }
        free_tensor(&pred);
    }

    float acc = (float)correct / test_samples * 100.0f;
    printf("\n========================================\n");
    printf("FINAL ACCURACY: %.2f%%\n", acc);
    printf("========================================\n");

    free_mnist_data(x_test, y_test, test_samples);
    free_network(&net);

    return 0;
}



Network* get_network(int n_features) {
    Network* network_created = create_network(n_features, MSE, SGD, LEARNING_RATE);
    
    network_add_layer(network_created, 256, RELU);
    network_add_layer(network_created, 128, RELU);
    network_add_layer(network_created, 64, RELU);
    
    network_add_layer(network_created, 10, LINEAR);

    return network_created;
}


void create_mini_batches(Tensor** x_in, Tensor** y_in, int total_samples, int batch_size, Tensor*** x_out, Tensor*** y_out, int* total_batches) {
    int n_batches = total_samples / batch_size;
    *total_batches = n_batches;
    int features = x_in[0]->cols;
    int classes = y_in[0]->cols;

    *x_out = (Tensor**)malloc(n_batches * sizeof(Tensor*));
    *y_out = (Tensor**)malloc(n_batches * sizeof(Tensor*));

    for (int b = 0; b < n_batches; b++) {
        Tensor* bx = create_tensor_value(batch_size, features, 0.0f);
        Tensor* by = create_tensor_value(batch_size, classes, 0.0f);

        for (int i = 0; i < batch_size; i++) {
            int src_idx = (b * batch_size) + i;
            
            // Use memcpy to copy the entire row at once
            memcpy(&(bx->data[i * features]), x_in[src_idx]->data, features * sizeof(float));
            memcpy(&(by->data[i * classes]), y_in[src_idx]->data, classes * sizeof(float));
        }

        (*x_out)[b] = bx;
        (*y_out)[b] = by;
    }
}



int get_predicted_class(Tensor* t) {
    if (!t) return -1;
    
    int max_idx = 0;
    float max_val = t->data[0];
    for (int i = 1; i < t->cols; i++) {
        if (t->data[i] > max_val) {
            max_val = t->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/* CSV Loader */
int load_mnist_csv(const char* filename, Tensor*** x_data, Tensor*** y_data, int* num_samples, int* n_features) {
    FILE* file = fopen(filename, "r");
    if (!file) { printf("Error opening %s\n", filename); return 0; }

    char line[10000];
    int row_count = 0;
    int cols = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) < 5) continue;
        if (row_count == 0 && !isdigit(line[0])) continue;
        
        if (cols == 0) {
            char* tmp = strdup(line);
            char* tok = strtok(tmp, ",");
            while(tok) { cols++; tok = strtok(NULL, ","); }
            free(tmp);
        }
        row_count++;
    }
    *n_features = cols - 1;
    *num_samples = row_count;

    rewind(file);
    *x_data = (Tensor**)malloc(row_count * sizeof(Tensor*));
    *y_data = (Tensor**)malloc(row_count * sizeof(Tensor*));
    int idx = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) < 5) continue;
        if (idx == 0 && !isdigit(line[0])) continue;

        char* token = strtok(line, ",");
        int label = atoi(token);

        (*y_data)[idx] = create_tensor_value(1, 10, 0.0f);
        if(label >=0 && label <= 9) (*y_data)[idx]->data[label] = 1.0f;

        (*x_data)[idx] = create_tensor_value(1, *n_features, 0.0f);
        for(int i=0; i<*n_features; i++) {
            token = strtok(NULL, ",");
            if(token) (*x_data)[idx]->data[i] = (float)atoi(token) / 255.0f;
        }
        idx++;
    }
    fclose(file);
    return 1;
}

void free_mnist_data(Tensor** x_data, Tensor** y_data, int count) {
    for (int i = 0; i < count; i++) {
        free_tensor(&(x_data[i]));
        free_tensor(&(y_data[i]));
    }
    free(x_data);
    free(y_data);
}
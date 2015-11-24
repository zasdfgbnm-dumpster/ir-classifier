#include <fann.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "parameters.h"

const unsigned int num_layers = 3;
int num_neurons_hidden;
const float desired_error = (const float) 0.001;
const unsigned int max_epochs = 20000;
const unsigned int epochs_between_reports = 100;

int main(int argc, char *argv[])
{
    // read num_neurons_hidden from arguments
    num_neurons_hidden = atoi(argv[1]);
    
    // setup neural network
    struct fann *ann = fann_create_standard(num_layers,num_input,num_neurons_hidden,num_output);
    fann_set_activation_function_hidden(ann,FANN_SIGMOID);
    fann_set_activation_function_output(ann,FANN_SIGMOID);
    
    // setup training set
    struct fann_train_data *data = fann_read_train_from_file("fann_data.txt");
    struct fann_train_data *train = fann_subset_train_data(data,0,num_train);
    puts("finish loading data set");
    
    // train
    fann_train_on_data(ann,train,max_epochs,epochs_between_reports,desired_error);
    
    // save
    char fn[200];
    sprintf(fn,"cooh-%d.net",num_neurons_hidden);
    fann_save(ann, fn);
    
    // test
    
    // end of program
    fann_destroy_train(data);
    fann_destroy_train(train);
    fann_destroy(ann);
    return 0;
}

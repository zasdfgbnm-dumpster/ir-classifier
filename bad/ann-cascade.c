#include <fann.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "parameters.h"

const float desired_error = (const float) 0.001;
unsigned int max_neurons = 100;
unsigned int neurons_between_reports = 1;

int main(int argc, char *argv[])
{
	
    // setup training set
    struct fann_train_data *data = fann_read_train_from_file("fann_data.txt");
    struct fann_train_data *train = fann_subset_train_data(data,0,num_train);
    puts("finish loading data set");

    // setup neural network
    struct fann *ann = fann_create_shortcut(2, num_input, num_output);
    
    // train
    fann_cascadetrain_on_data(ann,train,max_neurons,neurons_between_reports,desired_error);
    
    // save
    fann_save(ann, "cooh-cascade.net");
    
    // test
    
    // end of program
    fann_destroy_train(data);
    fann_destroy_train(train);
    fann_destroy(ann);
    return 0;
}

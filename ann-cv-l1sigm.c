#include "cv-train.h"

const unsigned int num_layers = 3;
unsigned int num_hidden_nodes;

struct fann *random_network(){
	// create and randomize activation function
	struct fann *ann = fann_create_standard(num_layers,num_input,num_hidden_nodes,num_output);
	fann_set_activation_function_hidden(ann,FANN_SIGMOID);
	fann_set_activation_function_output(ann,FANN_SIGMOID);
	
	// randomize weights
	fann_randomize_weights(ann,-1,1);
	
	// return
	return ann;
}

int main(int argc, char *argv[])
{
	num_hidden_nodes = atoi(argv[1]);
	doall(argv[1]);
	return 0;
}

#include "cv-train.h"

const unsigned int min_neurons_hidden = 2;
const unsigned int max_neurons_hidden = 10;
const unsigned int min_layers = 3;
const unsigned int max_layers = 5;

unsigned int random_hidden_activation(){
	unsigned int list[] = { FANN_LINEAR, FANN_SIGMOID, FANN_GAUSSIAN, FANN_ELLIOT, FANN_LINEAR_PIECE, FANN_SIN, FANN_COS };
	unsigned int n = sizeof(list)/sizeof(unsigned int);
	unsigned int r = rand()%n;
	return list[r];
}

unsigned int random_output_activation(){
	unsigned int list[] = { FANN_SIGMOID, FANN_ELLIOT, FANN_LINEAR_PIECE, FANN_SIN, FANN_COS };
	unsigned int n = sizeof(list)/sizeof(unsigned int);
	unsigned int r = rand()%n;
	return list[r];
}

struct fann *random_network(){
	// random number of layers and number of nodes per layer
	const unsigned int num_layers = randint(min_layers,max_layers);
	unsigned int num_nodes[num_layers];
	num_nodes[0] = num_input;
	for(int i=1;i<num_layers-1;i++){
		unsigned int num_neurons_hidden = randint(min_neurons_hidden,max_neurons_hidden);
		num_nodes[i] = num_neurons_hidden;
	}
	num_nodes[num_layers-1] = num_output;
	
	// create and randomize activation function
	struct fann *ann = fann_create_standard_array(num_layers,num_nodes);
	for(int i=1;i<num_layers-1;i++)
		for(int j=0;j<num_nodes[i];j++)
			fann_set_activation_function(ann,random_hidden_activation(),i,j);
	fann_set_activation_function_output(ann,random_output_activation());
	
	// randomize weights
	fann_randomize_weights(ann,-1,1);
	
	// return
	return ann;
}

int main(int argc, char *argv[])
{
	doall(argv[1]);
	return 0;
}

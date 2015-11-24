#include <fann.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "parameters.h"
#include "correct_rate.h"

void stat(char *annfile,unsigned int start, unsigned int n)
{
	// setup neural network
	struct fann *ann = fann_create_from_file(annfile);
	fann_reset_MSE(ann);
	
	// setup training set
	struct fann_train_data *data = fann_read_train_from_file("fann_data.txt");
	struct fann_train_data *subset = fann_subset_train_data(data,start,n);
	puts("finish loading data set");
	
	// test
	int countpp = 0;
	int countpn = 0;
	int countnp = 0;
	int countnn = 0;
	corret_rate(ann,subset,n,&countpp,&countpn,&countnp,&countnn,1);

	printf("\n");
	
	printf("++: %d\n",countpp);
	printf("+-: %d\n",countpn);
	printf("-+: %d\n",countnp);
	printf("--: %d\n",countnn);
	printf("\n");
	
	printf("+ correct rate: %f%%\n",100.0*countpp/(countpp+countpn));
	printf("- correct rate: %f%%\n",100.0*countnn/(countnp+countnn));
	
	// end of program
	fann_destroy_train(data);
	fann_destroy_train(subset);
	fann_destroy(ann);
}

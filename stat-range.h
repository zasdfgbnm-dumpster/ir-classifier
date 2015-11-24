#include <fann.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "parameters.h"

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
    for(int i=0;i<n;i++){
	    fann_type *input = subset->input[i];
	    int desired_output = *(subset->output[i])>0.5?1:0;
	    int output = *fann_run(ann,input)>0.5?1:0;
	    if(desired_output&&output)
		    countpp++;
	    if(desired_output&&!output){
		    printf("%d: desired 1 get 0\n",i);
		    countpn++;
	    }
	    if(!desired_output&&output){
		    printf("%d: desired 0 get 1\n",i);
		    countnp++;
	    }
	    if(!desired_output&&!output)
		    countnn++;
    }
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

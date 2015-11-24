#include <fann.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <sys/file.h>
#include <unistd.h>
#include "parameters.h"

const unsigned int max_epochs = 50000;
const unsigned int terminate_if_no_better = 5000;
const unsigned int epochs_between_reports = 1;
const unsigned int epochs_between_prints = 10;
const double desired_error = 0.000;
#define num_cross_validation (4)

struct fann_train_data *data = NULL;
struct fann_train_data *train = NULL;
struct fann_train_data *cv_test = NULL;

double best_test_error[num_cross_validation];
int best_epoch[num_cross_validation];
fann_type *weights[num_cross_validation];
int current_cv;

void init_data(){
	data = fann_read_train_from_file("fann_data.txt");
	train = fann_subset_train_data(data,0,num_train);
	puts("finish loading data set");
}

void cleanup(){
	fann_destroy_train(data);
	fann_destroy_train(train);
}

unsigned int randint(unsigned int l, unsigned int r){
	unsigned int ret = rand()%(r-l+1)+l;
	return ret;
}

void debug(char *s){
	puts(s);
	fflush(stdout);
}

double load_best(){
	double best_error;
	FILE *fp = fopen("best.txt","r");
	flock(fileno(fp),LOCK_SH);
	int ret1 = fscanf(fp,"%lf",&best_error);
	flock(fileno(fp),LOCK_UN);
	fclose(fp);
	return best_error;
}

struct fann *random_network();

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
                           unsigned int max_epochs, unsigned int epochs_between_reports,
                           float desired_error, unsigned int epochs)
{
	// run test on cross validation test set
	struct fann *cpy = fann_copy(ann);
	fann_reset_MSE(cpy);
	fann_test_data(cpy,cv_test);
	double error = fann_get_MSE(cpy);
	if(error < best_test_error[current_cv]){
		best_epoch[current_cv] = epochs;
		best_test_error[current_cv] = error;
		fann_get_weights(ann,weights[current_cv]);
	}
	fann_destroy(cpy);
	if(epochs%epochs_between_prints==0)
		printf("CV: %d, Epochs: %u, train error: %lf, test error: %lf, best error: %lf\n",current_cv,epochs,fann_get_MSE(ann),error,best_test_error[current_cv]);
	if(epochs-best_epoch[current_cv] > terminate_if_no_better){
		printf("no improve for %u epochs, terminate\n",terminate_if_no_better);
		return -1;
	}
}

double cross_validation(struct fann *ann, struct fann_train_data *data){
	double error = 0;
	// set callback
	fann_set_callback(ann,test_callback);
	// manipulate trainning data
	fann_shuffle_train_data(train);
	struct fann_train_data *train[num_cross_validation];
	struct fann_train_data *test[num_cross_validation];
	const unsigned int sz = num_train / num_cross_validation;
	for(int i=0;i<num_cross_validation;i++){
		test[i] = fann_subset_train_data(data,sz*i,sz);
		struct fann_train_data *first = fann_subset_train_data(data,0,sz*i);
		struct fann_train_data *second = fann_subset_train_data(data,sz*(i+1),sz*(num_cross_validation-i-1));
		train[i] = fann_merge_train_data(first,second);
		fann_destroy_train(first);
		fann_destroy_train(second);
	}
	
	// store initial condition
	unsigned int num_weights = fann_get_total_connections(ann);
	fann_type init_weights[num_weights];
	fann_get_weights(ann,init_weights);
	
	// do training
	double errors[num_cross_validation];
	for(int i=0;i<num_cross_validation;i++){
		best_test_error[i] = 9999;
		cv_test = test[i];
		current_cv = i;
		weights[i] = calloc(num_weights,sizeof(fann_type));
		// train
		fann_reset_MSE(ann);
		fann_set_weights(ann,init_weights);
		fann_train_on_data(ann,train[i],max_epochs,epochs_between_reports,desired_error);
		// test
		errors[i] = best_test_error[i];
	}
	
	// calculate average error
	error = 0;
	for(int i=0;i<num_cross_validation;i++)
		error += errors[i];
	error /= num_cross_validation;
	
	// if better put best cross validation weights to ann
	double best_error = load_best();
	if(error < best_error){
		int bestidx = 0;
		double besterr = 999;
		for(int i=0;i<num_cross_validation;i++){
			if(best_test_error[i]<besterr){
				bestidx = i;
				besterr = best_test_error[i];
			}
		}
		fann_set_weights(ann,weights[bestidx]);
	}
	
	// cleanup
	for(int i=0;i<num_cross_validation;i++){
		fann_destroy_train(train[i]);
		fann_destroy_train(test[i]);
		free(weights[i]);
	}
	
	// return
	return error;
}

void save(double err,struct fann *ann){
	// save best error
	FILE *fp = fopen("best.txt","r+");
	int fd = fileno(fp);
	flock(fd,LOCK_EX);
	
	double best_error;
	int ret1 = fscanf(fp,"%lf",&best_error);
	if(err<best_error){
		fseek(fp,0,SEEK_SET);
		int ret2 = ftruncate(fd,0);
		fprintf(fp,"%lf",err);
		fann_save(ann, "best.net");
	}
	
	flock(fd,LOCK_UN);
	fclose(fp);
}

void doall(){
	srand(time(NULL));
	init_data();
	
	struct fann *ann = random_network();
	fann_print_parameters(ann);
	fflush(stdout);
	
	double new_error = cross_validation(ann,train);
	save(new_error,ann);
	
	// end of program
	cleanup();
	fann_destroy(ann);
}
void corret_rate(struct fann *ann,struct fann_train_data *subset,int n,int *countpp, int *countpn, int *countnp, int *countnn,int print){
	(*countpp) = 0;
	(*countpn) = 0;
	(*countnp) = 0;
	(*countnn) = 0;
	for(int i=0;i<n;i++){
		fann_type *input = subset->input[i];
		int desired_output = *(subset->output[i])>0.5?1:0;
		int output = *fann_run(ann,input)>0.5?1:0;
		if(desired_output&&output)
			(*countpp)++;
		if(desired_output&&!output){
			if(print) printf("%d: desired 1 get 0\n",i);
			(*countpn)++;
		}
		if(!desired_output&&output){
			if(print) printf("%d: desired 0 get 1\n",i);
			(*countnp)++;
		}
		if(!desired_output&&!output)
			(*countnn)++;
	}
}
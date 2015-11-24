#include <fann.h>
#include <stdio.h>
#include <string.h>
#include "stat-range.h"

int main(int argc, char *argv[])
{
    stat(argv[1],num_train,num_dev-num_train);
}

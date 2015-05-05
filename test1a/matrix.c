#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(int size, double *matrix0, double *matrix1){
  int i;
  //for(i=0; i<size; i++)
    //vector[i] = ((double)rand())/65535.0;
  for(i=0; i<size*size; i++)
    matrix0[i] = ((double)rand())/5307.0;
  for(i=0; i<size*size; i++)
    matrix1[i] = ((double)rand())/5307.0;


}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, double *matrix0_in,
		       double *matrix1_in, double *matrix_out){
  int rows, cols,k;
  //int j;
  for(rows=0; rows<size; rows++){
    for(cols=0; cols<size; cols++){
      matrix_out[cols] = 0.0;
	for(k=0;k<size;k++)
      	matrix_out[cols+rows*size] += matrix0_in[rows*size + k ] * matrix1_in[k*size + cols];
    }
  }
}

void matrix_mult_pl(int size, double *matrix0_in,
		       double *matrix1_in, double *matrix_out){
  int rows, cols, k;
  //int j;
# pragma omp parallel				\
  shared(size, matrix0_in, matrix1_in, matrix_out)	\
  private(rows, cols)
# pragma omp for
  for(rows=0; rows<size; rows++){
    for(cols=0; cols<size; cols++){
      matrix_out[cols] = 0.0;
	for(k=0;k<size;k++)
      	matrix_out[cols+rows*size] += matrix0_in[rows*size + k ] * matrix1_in[k*size + cols];
    }
  }
}
int main(int argc, char *argv[]){
  if(argc < 2){
    printf("Usage: %s matrix_size\n", argv[0]);
    return 0;
  }

  int size = atoi(argv[1]);
  double *matrix0 = (double *)malloc(sizeof(double)*size*size);
  double *matrix1 = (double *)malloc(sizeof(double)*size*size);
  double *result_sq = (double *)malloc(sizeof(double)*size*size);
  double *result_pl = (double *)malloc(sizeof(double)*size*size);
  matrix_vector_gen(size, matrix0, matrix1);

  double time_sq = 0;
  double time_pl = 0;

  time_sq = omp_get_wtime();
  matrix_mult_sq(size, matrix0, matrix1, result_sq);
  time_sq = omp_get_wtime() - time_sq;

  time_pl = omp_get_wtime();
  matrix_mult_pl(size, matrix0, matrix1, result_pl);
  time_pl = omp_get_wtime() - time_pl;

  printf("SEQUENTIAL EXECUTION: %f (sec)\n", time_sq);
  printf("PARALLEL EXECUTION WITH %d (threads) ON %d (processors): %f (sec)\n",
	 omp_get_max_threads(), omp_get_num_procs(), time_pl);

  //check
  int i;
  for(i=0; i<size*size; i++)
    if(result_sq[i] != result_pl[i]){
      printf("wrong at position %d\n", i);
      return 0;
    }

  free(matrix0);
  free(matrix1);
  free(result_sq);
  free(result_pl);
  return 1;
}

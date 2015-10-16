#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <xmmintrin.h>
#include <time.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(int size, int newsize, double *matrix, double *vector){
  int i;
  int j;
  for(i=0; i<size; i++)
    vector[i] = i*1.2f + 1;//((float)rand())/65535.0f;
  for(i=size; i<newsize; i++)
    vector[i] = 0.0f;

  for(i=0; i<newsize*newsize; i++)
    matrix[i] = 0.0f;//((float)rand())/5307.0f;
  for(i=0;i<size; i++){
	for(j=0;j<size;j++){
	matrix[i*newsize+j] = (j + size*i) *1.3f + 1;
	} 	
  }
  
}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, double *vector_in,
		       double *matrix_in, double *vector_out){
  int rows, cols;
  int j;
  for(cols=0; cols<size; cols++){
    vector_out[cols] = 0.0;
    for(j=0,rows=0; rows<size; j++,rows++)
      vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
  }
}

void matrix_mult_sse(int size, double *vector_in,
		      double *matrix_in, double *vector_out){
  __m128d a_line, b_line, r_line;
  int i, j;
  for (i=0; i<size; i+=2){
    j = 0;
    b_line = _mm_load_pd(&matrix_in[i]); // b_line = vec4(matrix[i][0])
    a_line = _mm_set1_pd(vector_in[j]);      // a_line = vec4(vector_in[0])
    r_line = _mm_mul_pd(a_line, b_line); // r_line = a_line * b_line
    for (j=1; j<size; j++) {
      b_line = _mm_load_pd(&matrix_in[j*size+i]); // a_line = vec4(column(a, j))
      a_line = _mm_set1_pd(vector_in[j]);  // b_line = vec4(b[i][j])
                                     // r_line += a_line * b_line
      r_line = _mm_add_pd(_mm_mul_pd(a_line, b_line), r_line);
    }
    _mm_store_pd(&vector_out[i], r_line);     // r[i] = r_line
  }
}


int main(int argc, char *argv[]){
  if(argc < 2){
    printf("Usage: %s matrix/vector_size\n", argv[0]);
    return 0;
  }

  int size = atoi(argv[1]);
  int newsize = size;
  if(size%4 != 0){
      newsize = (size/4+1)*4;
      printf("Generate a new size\n");
}

  double *vector = (double *)memalign(sizeof(double)*2, sizeof(double)*newsize);//(float *)malloc(sizeof(float)*size);
  if(vector==NULL){
    printf("can't allocate the required memory for vector\n");
    return 0;
  }
  //printf("after vector generation");

  double *matrix = (double *)memalign(sizeof(double)*2, sizeof(double)*newsize*newsize);
  if(matrix==NULL){
    printf("can't allocate the required memory for matrix\n");
    free(vector);
    return 0;
  }

  double *result_sq = (double *)memalign(sizeof(double), sizeof(double)*size);
  if(result_sq==NULL){
    printf("can't allocate the required memory for result_sq\n");
    free(vector);
    free(matrix);
    return 0;
  }

  double *result_pl = (double *)memalign(sizeof(double)*2, sizeof(double)*newsize);
  if(result_pl==NULL){
    printf("can't allocate the required memory for result_pl\n");
    free(vector);
    free(matrix);
    free(result_sq);
    return 0;
  }

  matrix_vector_gen(size, newsize, matrix, vector);


  double time_sq;
  double time_sse;

  time_sq = omp_get_wtime();
  matrix_mult_sq(size, vector, matrix, result_sq);
  time_sq = omp_get_wtime() - time_sq;


  time_sse = omp_get_wtime();
  matrix_mult_sse(newsize, vector, matrix, result_pl);
  time_sse = omp_get_wtime() - time_sse;


  printf("SEQUENTIAL EXECUTION: %f (sec)\n",time_sq);
  printf("PARALLEL EXECUTION: %f (sec)\n", time_sse);

  //check
  /*int i;
  for(i=0; i<size; i++)
    if((int)result_sq[i] != (int)result_pl[i]){
      printf("wrong at position %d\n", i);
      free(vector);
      free(matrix);
      free(result_sq);
      free(result_pl);
      return 0;
    }*/

  free(vector);
  free(matrix);
  free(result_sq);
  free(result_pl);
  return 1;
}

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
void matrix_gen(int size, int newsize, float *matrix0, float *matrix1){
  int i;
  int j;
  // for(i=0; i<size; i++)
  //   vector[i] = i*1.2f + 1;//((float)rand())/65535.0f;
  // for(i=size; i<newsize; i++)
  //   vector[i] = 0.0f;

  for(i=0; i<newsize*newsize; i++)
    matrix0[i] = 0.0f;//((float)rand())/5307.0f;
    matrix1[i] = 0.0f;
  for(i=0;i<size; i++){
	 for(j=0;j<size;j++){
	 matrix0[i*newsize+j] = (j + size*i) *1.3f + 1;
   matrix1[i*newsize+j] = (j + size*i) *1.4f + 1;
	 } 	
  }
  
}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, float *matrix0_in,
		       float *matrix1_in, float *matrix_out){
  int rows, cols,k;
  // int j;
  // for(cols=0; cols<size; cols++){
  //   vector_out[cols] = 0.0;
  //   for(j=0,rows=0; rows<size; j++,rows++)
  //     vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
  // }
  for(rows=0; rows<size; rows++){
    for(cols=0; cols<size; cols++){
      matrix_out[cols] = 0.0;
      for(k=0;k<size;k++)
        matrix_out[cols+rows*size] += matrix0_in[rows*size + k ] * matrix1_in[k*size + cols];
    }
  }
}

void matrix_mult_sse(int size, float *matrix0_in,
		      float *matrix1_in, float *matrix_out){
  __m128 a_line, b_line, r_line;
  int i, j, k;
  for(k =0; k<size; k++){  
    	for (i=0; i<size; i+=4){
      	j = 0;
      	b_line = _mm_load_ps(&matrix1_in[i]); // b_line = vec4(matrix[i][0])
      	a_line = _mm_set1_ps(matrix0_in[j+size*k]);      // a_line = vec4(vector_in[0])
      	r_line = _mm_mul_ps(a_line, b_line); // r_line = a_line * b_line
      	for (j=1; j<size; j++) {
        	b_line = _mm_load_ps(&matrix1_in[j*size+i]); // a_line = vec4(column(a, j))
        	a_line = _mm_set1_ps(matrix0_in[j+size*k]);  // b_line = vec4(b[i][j])
                                     // r_line += a_line * b_line
        	r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
      	}
      	_mm_store_ps(&matrix_out[i+size*k], r_line);     // r[i] = r_line
    	}  
  }	
}


int main(int argc, char *argv[]){
  if(argc < 2){
    printf("Usage: %s matrix1/matrix0\n", argv[0]);
    return 0;
  }

  int size = atoi(argv[1]);
  int newsize = size;
  if(size%4 != 0){
      newsize = (size/4+1)*4;
      printf("Generate a new size\n");
}

  float *matrix0 = (float *)memalign(sizeof(float)*4, sizeof(float)*newsize*newsize);//(float *)malloc(sizeof(float)*size);
  if(matrix0==NULL){
    printf("can't allocate the required memory for matrix0\n");
    return 0;
  }
  //printf("after vector generation");

  float *matrix1 = (float *)memalign(sizeof(float)*4, sizeof(float)*newsize*newsize);
  if(matrix1==NULL){
    printf("can't allocate the required memory for matrix1\n");
    free(matrix0);
    return 0;
  }

  float *result_sq = (float *)memalign(sizeof(float), sizeof(float)*size*size);
  if(result_sq==NULL){
    printf("can't allocate the required memory for result_sq\n");
    free(matrix0);
    free(matrix1);
    return 0;
  }

  float *result_pl = (float *)memalign(sizeof(float)*4, sizeof(float)*newsize*newsize);
  if(result_pl==NULL){
    printf("can't allocate the required memory for result_pl\n");
    free(matrix0);
    free(matrix1);
    free(result_sq);
    return 0;
  }

  matrix_gen(size, newsize, matrix1, matrix0);


  double time_sq;
  double time_sse;

  time_sq = omp_get_wtime();
  matrix_mult_sq(size, matrix0, matrix1, result_sq);
  time_sq = omp_get_wtime() - time_sq;


  time_sse = omp_get_wtime();
  matrix_mult_sse(newsize, matrix0, matrix1, result_pl);
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

  free(matrix0);
  free(matrix1);
  free(result_sq);
  free(result_pl);
  return 1;
}

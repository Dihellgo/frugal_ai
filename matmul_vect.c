
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <tgmath.h>

#include <sys/time.h>

struct timeval timer;

void init_timing(struct timeval* tstart)
{
	gettimeofday(tstart, NULL);
}

float elapsed_time(struct timeval tstart)
{
	struct timeval tmp;
	long long diff;
	gettimeofday(&tmp, NULL);
	diff = tmp.tv_usec - tstart.tv_usec;
	diff += (tmp.tv_sec - tstart.tv_sec) * 1000000;
	return ((float)diff*1.0e-6);
}

void matmul_blas(const float* A, const float* B, float* C, int M, int N, int K)
{
    float alpha = 1.0f, beta = 0.0f;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, alpha, A, M, B, K, beta, C, M);
}

void create_matrices(float** A, float** B, float** C, float** C_blas, int M, int N, int K)
{
    size_t i;

    *A = (float*) calloc(M*K,sizeof(float));
    *B = (float*) calloc(K*N,sizeof(float));
    *C = (float*) calloc(M*N,sizeof(float));
    *C_blas = (float*) calloc(M*N,sizeof(float));

    for(i = 0; i < M*K; i++)
        (*A)[i] = ((float)rand()/RAND_MAX-0.5)*0.1;

    for(i = 0; i < K*N; i++)
        (*B)[i] = ((float)rand()/RAND_MAX-0.5)*0.1;
}


//v3 + manual vectorization through memory aligned SIMD operations 
//Note, require C11 ISO, not compatible with C99 while auto vectorization is compatible

// Define a memory aligned vector type of 32 Bytes -> 8 float of 4 bytes (32 bits) each
typedef float vec __attribute__ (( vector_size(32) ));

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
	if(K % 8 != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}
	
	// Some vector creation and usage exemples
	// Dynamically allocate a space in memory composed of X elements using the size of one element as alignement
	// vec *a = (vec*) aligned_alloc(sizeof(vec),X*sizeof(vec));  
	
	// Allocate and initialize a vector to a constant value
	// vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	
	// Fill a vector through indexing in a loop
	// for (i = 0; i < 8; i++)
	//     v[i] = i;
	
	// Access element i of a vectorized version of the matrix A
	// a[i/8][i%8] = A[i];
	
	int vec_K = K/8;	
	
	//1° Create the vectorized versions of A and B
	vec *At = (vec*) aligned_alloc(sizeof(vec),vec_K*M*sizeof(vec));  
	vec *Bvec = (vec*) aligned_alloc(sizeof(vec),vec_K*N*sizeof(vec));  
	
	//2° Fill the vectorized versions
	for (int i=0;i < M;i++) {
	    for (int k=0;k < vec_K; k++) {
	        for (int j=0;j < 8;j++) {
		    At[i*vec_K + k][j] = A[(k*8+j)*M+i];
		}
    	    }
	}

	for (int j=0;j<N;j++) {
	    for (int k=0;k<vec_K;k++) {
	    	for (int i=0;i<8;i++) {
	    	   Bvec[j*vec_K + k][i] = B[j*K + k*8 + i];
		}
	    }
	}	    
	
	//3° Do the i,j,k loops with the vectorized accumulate loop over k and fill the corresponding cell of C
	for (int i=0;i<M;i++) {
	    for (int j=0;j<N;j++) {
		vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		for (int k=0;k<vec_K;k++) {
		    acc = acc + At[i*vec_K + k]*Bvec[j*vec_K + k];
		}
		for (int k=0; k<8; k++) {
		    C[j*M +i] = C[j*M +i] + acc[k];
		}
	    }
	}
}



//Matmul using FMA operation on a sub-part of C and optimizing the cache usage
//No need for transposition in this version as most of the work is done in cache anyway

#define ker_w 6
#define ker_h 16

void kernel(const float *_a, const float *_b, float *_c, 
	int M, int N, int K, int i0, int j0, int k_start, int k_stop) 
{
	int k, b_m, b_n, l;
	float val;
	//declared in cache
	vec c_vec[ker_h/8][ker_w] = {};

	//1° Loop over k that do the vectorized accumulate inside the kernel region using an implicit fma instruction
	for (int k=k_start;k<k_stop;k++) {
	    // 1.1 Loop over ker_w
	    for (int j=0;j<ker_w;j++) {
		// Broadcast the b[k] value into a beta vector
		vec beta;
		for (int t=0;t<8;t++) {
		    beta[t] = _b[(j0+j)*K+k];
		}
		vec alpha;
		// 1.2 Loop over the vectors in A (ker_h/8)
		for (int t=0;t<ker_h/8;t++) {
		    // Fill an alpha vector with the content of the current vector of A
		    for (int t2=0;t2<8;t2++) {
			alpha[t2] = _a[k*M + i0+t*8+t2];
		    }
		    // Accumulate alpha * beta into the current vector cell of t[][]. Doing so with vectors will implicitely call an FMA operation.
		    c_vec[t][j] = c_vec[t][j] + alpha*beta;
		}
             }
        }	    
	//2° accumulate the result in the corresponding region of C
	for (int j=0;j<ker_w;j++) {
	    for (int i=0;i<ker_h;i++) {
		// printf("%f\n",_c[(j+j0)*M+i+i0]);
		// printf("%f\n",c_vec[i/8][j][i%8]);

		_c[(j+j0)*M+i+i0] = _c[(j+j0)*M+i+i0] + c_vec[i/8][j][i%8];
	    }
	}
}


void matmul_v5(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	if(M % ker_h != 0 || N % ker_w != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	//Simple loop over M and N kernel regions that call the kernel over the whole K dimension (k_start = 0, k_stop = K)
	
	for (int i=0; i<M/ker_h;i++) {
	    for (int j=0; j<N/ker_w;j++) {
	        kernel(A,B,C,M,N,K,i*ker_h,j*ker_w,0,K);
            }
	}

}



//V4 but with blocking decomposition to successively load sub parts of A and B
//into L2 and L1 cache for maximum reuse. Maximizing the achievable memory bandwith

#define l3 64 //Number of rows from A
#define l2 240 //Number of columns from B
#define l1 320 //Number of columns from A
void matmul_v6(const float *A, const float *B, float *C, int M, int N, int K) 
{
	
	// Set the size of the blocks and order them by importance into the different cache levels
	// Keep in mind that we want l3 < l2 < l1 and ensure that each dimension is a multiple of the corresponding kernel size
	
	if(K % l1 != 0 || M % l3 != 0 || N % l2 != 0 || l2 % ker_w != 0 || l3 % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	//5-nested loop version that goes through the blocks in cache and then call the kernel on the sub regions in the current block
	#pragma omp parallel for schedule(dynamic)
	for (int i1=0;i1<M/l3;i1++) {
	    for (int i2=0;i2<N/l2;i2++) {
		for (int i3=0;i3<K/l1;i3++) {
	            for (int i4=0;i4<l3/ker_h;i4++) {
			for (int i5=0;i5<l2/ker_w;i5++) {
			    kernel(A,B,C,M,N,K,i1*l3+i4*ker_h,i2*l2+i5*ker_w,i3*l1, (i3+1)*l1);
			}
		    }
		}
	    }
	}
}


int main(int argc, char *argv[])
{
	size_t i;
	
	int l_size, M, N, K;
	float *A, *B, *C, *C_blas;

	if (argc == 2)
		l_size = atoi(argv[1]);
	else
		l_size = 1920;
	
	M = l_size; N = l_size; K = l_size;

	create_matrices(&A, &B, &C, &C_blas, M, N, K);

	init_timing(&timer);

	matmul_blas(A,B,C_blas,M,N,K);

	printf("Regular sgemm time: %f s\n", elapsed_time(timer));

	init_timing(&timer);

	matmul_v6(A,B,C,M,N,K); //your custom matmul implementation
	
	printf("Custom sgemm time: %f s\n", elapsed_time(timer));
	
	//Verify the content of C against the OpenBLAS matmul
	//for (i = 0; i < M*N; i++)
	//	if((C[i] - C_blas[i])*(C[i] - C_blas[i]) > 0.00001)
	//	{
	//		printf("ERROR ! MATRIX DIFF !\n");
	//		printf("i:%ld C:%f C_blas:%f\n", i, C[i], C_blas[i]);
	//		exit(EXIT_FAILURE);
	//	}
	exit(EXIT_SUCCESS);
}
		
		

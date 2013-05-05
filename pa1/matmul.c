#include <stdio.h>
#include <xmmintrin.h>

/*
	Multiple 2x2 blocks using SSE2 instrinsics.
	TODO comments.
*/
inline matmul2x2(int N, const double *A, const double *B, double *C, int i, int j, int k, int k_max) {

	// 128-bit MMX registers (each holds two doubles)
	__m128d a11, a12, a21, a22, b1, b2, c1, c2;
	
	c1 = _mm_load_pd(&C[i*N+0+j]);			
	c2 = _mm_load_pd(&C[i*N+N+j]);	
	
	for(k = k; k < k_max; k += 2) {
		a11 = _mm_load1_pd(&A[(i+0)*N+(k+0)]);
		a12 = _mm_load1_pd(&A[(i+0)*N+(k+1)]);
		a21 = _mm_load1_pd(&A[(i+1)*N+(k+0)]);
		a22 = _mm_load1_pd(&A[(i+1)*N+(k+1)]);
		
		b1 = _mm_load_pd(&B[(k+0)*N+j]);
		b2 = _mm_load_pd(&B[(k+1)*N+j]);
		
		a11 = _mm_mul_pd(a11, b1);
		a12 = _mm_mul_pd(a12, b2);
		a21 = _mm_mul_pd(a21, b1);
		a22 = _mm_mul_pd(a22, b2);
			
		c1 = _mm_add_pd(c1, a11);
		c1 = _mm_add_pd(c1, a12);
		c2 = _mm_add_pd(c2, a21);
		c2 = _mm_add_pd(c2, a22);		

				
	}
	_mm_store_pd(&C[(i+0)*N+j], c1);
	_mm_store_pd(&C[(i+1)*N+j], c2);
	
}

/*
	Blocking.
	SSE2 intrinsics & block_size 32 (32 for N > 64)
*/
void matmul(int N, const double* A, const double* B, double* C) {
	int i, j, k;
	int ii, jj, kk;
	int B1 = 8;
	int ib_max, jb_max, kb_max;
	
	/*
	printf("size of float: %d\n", sizeof(float));
	printf("size of double: %d\n", sizeof(double));
	printf("size of long double: %d\n", sizeof(long double));
	*/
	
	if (N <= B1){
		for(i = 0; i < N; i+=2)  
			for(j = 0; j < N; j+=2)
				matmul2x2(N, A, B, C, i, j, 0, N);
				
	} else {
		for(i = 0; i < N; i+=B1)
			for(j = 0; j < N; j+=B1) {
				
				__builtin_prefetch();
			
				for(k = 0, ib_max=(i+B1<N)?(i+B1):N; k < N; k+=B1)
					for (ii = i, jb_max=(j+B1<N)?(j+B1):N; ii<ib_max; ii+=2)
						for (jj = j, kb_max=(k+B1<N)?(k+B1):N; jj<jb_max; jj+=2)							
							matmul2x2(N, A, B, C, ii, jj, k, kb_max);
			}				
	}
}

/*
	
	coherency_line_size: 64
	number_of_sets: 64
	size: 32K
	ways_of_associativity : 8
	level: 1
	type: Data
*/	


/*	SAVE THIS FOR FUTURE REFERENCE...
void matmul(int N, const double* A, const double* B, double* C) {
	int i, j, k;
	int ii, jj, kk;
	int B1 = 32;
	int ib_max, jb_max, kb_max;

	__m128d a11, a12, a21, a22;
	__m128d b1, b2;
	__m128d c1, c2;
	
	if (N <= 64){
		for(i = 0; i < N; i+=2)  
		for(j = 0; j < N; j+=2) {

			matmul2x2(N, A, B, C, i, j, 0, N);
			
			c1 = _mm_load_pd(&C[i*N+0+j]);			
			c2 = _mm_load_pd(&C[i*N+N+j]);
			
			for(k = 0; k < N; k+=2) {

				a11 = _mm_load1_pd(&A[i*N+0+k+0]);
				a12 = _mm_load1_pd(&A[i*N+0+k+1]);
				a21 = _mm_load1_pd(&A[i*N+N+k+0]);
				a22 = _mm_load1_pd(&A[i*N+N+k+1]);

				b1 = _mm_load_pd(&B[k*N+0+j]);
				b2 = _mm_load_pd(&B[k*N+N+j]);

				a11 = _mm_mul_pd(a11, b1);
				a12 = _mm_mul_pd(a12, b2);
				a21 = _mm_mul_pd(a21, b1);
				a22 = _mm_mul_pd(a22, b2);
			
				c1 = _mm_add_pd(c1, a11);
				c1 = _mm_add_pd(c1, a12);
				c2 = _mm_add_pd(c2, a21);
				c2 = _mm_add_pd(c2, a22);								
				
				//C[i*N+j]+=A[i*N+k]*B[k*N+j];
			}
			_mm_store_pd(&C[i*N+0+j], c1);
			_mm_store_pd(&C[i*N+N+j], c2);
		}
	} else {
		for(i = 0; i < N; i+=B1) {
			for(j = 0; j < N; j+=B1) {	
				for(k = 0, ib_max=(i+B1<N)?(i+B1):N; k < N; k+=B1) {
					for (ii = i, jb_max=(j+B1<N)?(j+B1):N; ii<ib_max; ii+=2){
						for (jj = j, kb_max=(k+B1<N)?(k+B1):N; jj<jb_max; jj+=2){

							c1 = _mm_load_pd(&C[ii*N+0+jj]);			
							c2 = _mm_load_pd(&C[ii*N+N+jj]);
							for (kk = k; kk<kb_max; kk+=2){
								a11 = _mm_load1_pd(&A[ii*N+0+kk+0]);
								a12 = _mm_load1_pd(&A[ii*N+0+kk+1]);
								a21 = _mm_load1_pd(&A[ii*N+N+kk+0]);
								a22 = _mm_load1_pd(&A[ii*N+N+kk+1]);
				
								b1 = _mm_load_pd(&B[kk*N+0+jj]);
								b2 = _mm_load_pd(&B[kk*N+N+jj]);
				
								a11 = _mm_mul_pd(a11, b1);
								a12 = _mm_mul_pd(a12, b2);
								a21 = _mm_mul_pd(a21, b1);
								a22 = _mm_mul_pd(a22, b2);
							
								c1 = _mm_add_pd(c1, a11);
								c1 = _mm_add_pd(c1, a12);
								c2 = _mm_add_pd(c2, a21);
								c2 = _mm_add_pd(c2, a22);			
							}
								_mm_store_pd(&C[ii*N+0+jj], c1);
								_mm_store_pd(&C[ii*N+N+jj], c2);
							
						}
					}
				}
			}
		}
	}
}

*/

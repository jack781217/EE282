// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include <stdio.h>
#include <xmmintrin.h>

/*
	#0: the original code; no optimization 
*/
void matmul0(int N, const double* A, const double* B, double* C) {
	int i, j, k;
	int ib, jb, kb;
  
	for(i = 0; i < N; i++)  
		for(j = 0; j < N; j++)  
			for(k = 0; k < N; k++) 
				C[i*N+j]+=A[i*N+k]*B[k*N+j];
}

/*
	#1: blocking
*/
void matmul1(int N, const double* A, const double* B, double* C) {
	int i, j, k;
	int ib_max, jb_max, kb_max;
	int ib, jb, kb;

	int B1 = 32;
 
	if(N <= B1) {
		for(i = 0; i < N; i++)  
			for(j = 0; j < N; j++)  
				for(k = 0; k < N; k++) 
					C[i*N+j]+=A[i*N+k]*B[k*N+j];

	} else {

		if(N > 64) 
			B1 = 8;

		for(i = 0; i < N; i += B1)  {
			for(j = 0; j < N; j += B1)  {
				for(k = 0; k < N; k += B1) {
					ib_max=(i+B1<N)?(i+B1):N;
					for(ib=i; ib<ib_max; ib++) {
						jb_max=(j+B1<N)?(j+B1):N;
						for(jb=j;jb<jb_max;jb++) {
							kb_max=(k+B1<N)?(k+B1):N;
							for(kb=k; kb<kb_max; kb++){
								C[ib*N+jb]+=A[ib*N+kb]*B[kb*N+jb];
							}
						}
					}
				}
			}
		}
	}


}

/*
	SSE2 intrinsics & block_size 2
*/
void matmul(int N, const double* A, const double* B, double* C) {
	int i, j, k;
	int ii, jj, kk;
	int b = 2;

	__m128d a11, a12, a21, a22;
	__m128d b1, b2;
	__m128d c1, c2;

	for(i = 0; i < N; i+=b)  
		for(j = 0; j < N; j+=b) {

			c1 = _mm_load_pd(&C[i*N+0+j]);			
			c2 = _mm_load_pd(&C[i*N+N+j]);
			
			for(k = 0; k < N; k+=b) {

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

	
		
}


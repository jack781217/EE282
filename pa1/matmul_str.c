#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>


void madd(int N, 
          int Xpitch, const double X[], 
          int Ypitch, const double Y[],
          int Spitch, double S[]) {
  int i, j;
  int ii, jj, ib_max, jb_max;
  __m128d x, y, z;
  int Bs = 32;
 if (N <= Bs){
   for ( i = 0; i < N; i++){
    for (j = 0; j < N; j+=2){
   	  x = _mm_load_pd(&X[i*Xpitch+j]);
   	  y = _mm_load_pd(&Y[i*Ypitch+j]);
   	  z = _mm_add_pd(x, y);
   	  _mm_store_pd(&S[i*Spitch+j], z);
    }
  }
 }
  else {
  for (i = 0; i < N; i+= Bs){
  	for (j = 0; j < N; j+= Bs){
	   ib_max=(i+Bs<N)?(i+Bs):N;
	  for ( ii = i; ii < ib_max; ii++){
	  	jb_max=(j+Bs<N)?(j+Bs):N;
    	for (jj = j; jj < jb_max; jj+=2){
	   	  x = _mm_load_pd(&X[ii*Xpitch+jj]);
	   	  y = _mm_load_pd(&Y[ii*Ypitch+jj]);
	   	  z = _mm_add_pd(x, y);
	   	  _mm_store_pd(&S[ii*Spitch+jj], z);
	    }	
	  }  	
    }
  }	
 }
}

void msub(int N, 
          int Xpitch, const double X[], 
          int Ypitch, const double Y[],
          int Spitch, double S[]) {
  int i, j;
  int ii, jj, ib_max, jb_max;
  __m128d x, y, z;
  int Bs = 32;
 if (N <= Bs){
   for ( i = 0; i < N; i++){
    for (j = 0; j < N; j+=2){
   	  x = _mm_load_pd(&X[i*Xpitch+j]);
   	  y = _mm_load_pd(&Y[i*Ypitch+j]);
   	  z = _mm_sub_pd(x, y);
   	  _mm_store_pd(&S[i*Spitch+j], z);
    }
  }
 }
  else {
  for (i = 0; i < N; i+= Bs){
  	for (j = 0; j < N; j+= Bs){
	   ib_max=(i+Bs<N)?(i+Bs):N;
	  for ( ii = i; ii < ib_max; ii++){
	  	jb_max=(j+Bs<N)?(j+Bs):N;
    	for (jj = j; jj < jb_max; jj+=2){
	   	  x = _mm_load_pd(&X[ii*Xpitch+jj]);
	   	  y = _mm_load_pd(&Y[ii*Ypitch+jj]);
	   	  z = _mm_sub_pd(x, y);
	   	  _mm_store_pd(&S[ii*Spitch+jj], z);
	    }	
	  }  	
    }
  }	
 }
}


void mmult(int N, 
           int Xpitch, const double X[], 
           int Ypitch, const double Y[],
           int Zpitch, double Z[]) {
    
	int i, j, k;
	__m128d a11, a12, a21, a22;
	__m128d b1, b2;
	__m128d c1, c2;
	
     for(i = 0; i < N; i+=2){
		for(j = 0; j < N; j+=2) {
			c1 = _mm_load_pd(&Z[i*Zpitch+0+j]);			
			c2 = _mm_load_pd(&Z[(i+1)*Zpitch+j]);
			
			for(k = 0; k < N; k+=2) {

				a11 = _mm_load1_pd(&X[i*Xpitch+0+k+0]);
				a12 = _mm_load1_pd(&X[i*Xpitch+0+k+1]);
				a21 = _mm_load1_pd(&X[(i+1)*Xpitch+k+0]);
				a22 = _mm_load1_pd(&X[(i+1)*Xpitch+k+1]);

				b1 = _mm_load_pd(&Y[k*Ypitch+0+j]);
				b2 = _mm_load_pd(&Y[(k+1)*Ypitch+j]);

				a11 = _mm_mul_pd(a11, b1);
				a12 = _mm_mul_pd(a12, b2);
				a21 = _mm_mul_pd(a21, b1);
				a22 = _mm_mul_pd(a22, b2);
			
				c1 = _mm_add_pd(c1, a11);
				c1 = _mm_add_pd(c1, a12);
				c2 = _mm_add_pd(c2, a21);
				c2 = _mm_add_pd(c2, a22);								
			}
			_mm_store_pd(&Z[i*Zpitch+0+j], c1);
			_mm_store_pd(&Z[(i+1)*Zpitch+j], c2);
		}
    }
  }
  
  void mmult_noadd(int N, 
           int Xpitch, const double X[], 
           int Ypitch, const double Y[],
           int Zpitch, double Z[]) {
	int i, j, k;
	__m128d a11, a12, a21, a22;
	__m128d b1, b2;
	__m128d c1, c2;
	double zero[2];
	zero[0] = 0.0;
	zero[1] = 0.0;
	
     for(i = 0; i < N; i+=2){
		for(j = 0; j < N; j+=2) {
			c1 = _mm_load_pd(&zero[0]);	
			c2 = _mm_load_pd(&zero[0]);			

			for(k = 0; k < N; k+=2) {

				a11 = _mm_load1_pd(&X[i*Xpitch+0+k+0]);
				a12 = _mm_load1_pd(&X[i*Xpitch+0+k+1]);
				a21 = _mm_load1_pd(&X[(i+1)*Xpitch+k+0]);
				a22 = _mm_load1_pd(&X[(i+1)*Xpitch+k+1]);

				b1 = _mm_load_pd(&Y[k*Ypitch+0+j]);
				b2 = _mm_load_pd(&Y[(k+1)*Ypitch+j]);

				a11 = _mm_mul_pd(a11, b1);
				a12 = _mm_mul_pd(a12, b2);
				a21 = _mm_mul_pd(a21, b1);
				a22 = _mm_mul_pd(a22, b2);
			
				c1 = _mm_add_pd(c1, a11);
				c1 = _mm_add_pd(c1, a12);
				c2 = _mm_add_pd(c2, a21);
				c2 = _mm_add_pd(c2, a22);								
			}
			_mm_store_pd(&Z[i*Zpitch+0+j], c1);
			_mm_store_pd(&Z[(i+1)*Zpitch+j], c2);
		}
    }
  }
//
// Volker Strassen algorithm for matrix multiplication.
// Theoretical Runtime is O(N^2.807).
// Assume NxN matrices where N is a power of two.
// Algorithm:
//   Matrices X and Y are split into four smaller
//   (N/2)x(N/2) matrices as follows:
//          _    _          _   _
//     X = | A  B |    Y = | E F |
//         | C  D |        | G H |
//          -    -          -   -
//   Then we build the following 7 matrices (requiring
//   seven (N/2)x(N/2) matrix multiplications -- this is
//   where the 2.807 = log2(7) improvement comes from):
//     P0 = A*(F - H);
//     P1 = (A + B)*H
//     P2 = (C + D)*E
//     P3 = D*(G - E);
//     P4 = (A + D)*(E + H)
//     P5 = (B - D)*(G + H)
//     P6 = (A - C)*(E + F)
//   The final result is
//        _                                            _
//   Z = | (P3 + P4) + (P5 - P1)   P0 + P1              |
//       | P2 + P3                 (P0 + P4) - (P2 + P6)|
//        -                                            -
//
void mmult_fast(int N, 
                int Xpitch, const double X[], 
                int Ypitch, const double Y[],
                int Zpitch, double Z[]) {
  if (N <= 16) {
    mmult_noadd(N, Xpitch, X, Ypitch, Y, Zpitch, Z);
    return;
  }

  const int n = N/2;      // size of sub-matrices

  const double *A = X;    // A-D matrices embedded in X
  const double *B = X + n;
  const double *C = X + n*Xpitch;
  const double *D = C + n;

  const double *E = Y;    // E-H matrices embeded in Y
  const double *F = Y + n;
  const double *G = Y + n*Ypitch;
  const double *H = G + n;

  double *P[7];   // allocate temp matrices off heap
  const int sz = n*n*sizeof(double);
  int i;
  for (i = 0; i < 7; i++)
    P[i] = (double *) malloc(sz);
  double *T = (double *) malloc(sz);
  double *U = (double *) malloc(sz);

  // P0 = A*(F - H);
  msub(n, Ypitch, F, Ypitch, H, n, T);
  mmult_fast(n, Xpitch, A, n, T, n, P[0]);
  
  // P1 = (A + B)*H
  madd(n, Xpitch, A, Xpitch, B, n, T);
  mmult_fast(n, n, T, Ypitch, H, n, P[1]);

  // P2 = (C + D)*E
  madd(n, Xpitch, C, Xpitch, D, n, T);
  mmult_fast(n, n, T, Ypitch, E, n, P[2]);

  // P3 = D*(G - E);
  msub(n, Ypitch, G, Ypitch, E, n, T);
  mmult_fast(n, Xpitch, D, n, T, n, P[3]);

  // P4 = (A + D)*(E + H)
  madd(n, Xpitch, A, Xpitch, D, n, T);
  madd(n, Ypitch, E, Ypitch, H, n, U);
  mmult_fast(n, n, T, n, U, n, P[4]);

  // P5 = (B - D)*(G + H)
  msub(n, Xpitch, B, Xpitch, D, n, T);
  madd(n, Ypitch, G, Ypitch, H, n, U);
  mmult_fast(n, n, T, n, U, n, P[5]);

  // P6 = (A - C)*(E + F)
  msub(n, Xpitch, A, Xpitch, C, n, T);
  madd(n, Ypitch, E, Ypitch, F, n, U);
  mmult_fast(n, n, T, n, U, n, P[6]);

  // Z upper left = (P3 + P4) + (P5 - P1)
  madd(n, n, P[4], n, P[3], n, T);
  msub(n, n, P[5], n, P[1], n, U);
  madd(n, n, T, n, U, Zpitch, Z);

  // Z lower left = P2 + P3
  madd(n, n, P[2], n, P[3], Zpitch, Z + n*Zpitch);

  // Z upper right = P0 + P1
  madd(n, n, P[0], n, P[1], Zpitch, Z + n);
  
  // Z lower right = (P0 + P4) - (P2 + P6)
  madd(n, n, P[0], n, P[4], n, T);
  madd(n, n, P[2], n, P[6], n, U);
  msub(n, n, T, n, U, Zpitch, Z + n*(Zpitch + 1));

  free(U);   //deallocate temp matrices
  free(T);

  for (i = 0; i <7; i++){
  	    free(P[i]);
  }
}

void matmul(int N, const double* A, const double* B, double* C) {
    if (N <=16){
    mmult(N, N, A, N, B, N, C);
    }
    else{
   	double *D = (double*) malloc(N*N*sizeof(double));
	mmult_fast(N, N, A, N, B, N, D);
	madd(N, N, C, N, D, N, C);
	free(D);	
    }

}
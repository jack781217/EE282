#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>

inline mmult_2x2 (const double* A, const double* B, double* C){
	__m128d a11, a12, a21, a22, b1, b2, c1, c2;
	
 		c1 = _mm_load_pd(C);			
		c2 = _mm_load_pd(C+2);
		a11 = _mm_load1_pd(A);
		a12 = _mm_load1_pd(A+1);
		a21 = _mm_load1_pd(A+2);
		a22 = _mm_load1_pd(A+3);
				
		b1 = _mm_load_pd(B);
		b2 = _mm_load_pd(B+2);
				
		a11 = _mm_mul_pd(a11, b1);
		c1 = _mm_add_pd(c1, a11);
		a12 = _mm_mul_pd(a12, b2);
		c1 = _mm_add_pd(c1, a12);
		a21 = _mm_mul_pd(a21, b1);
		c2 = _mm_add_pd(c2, a21);
		a22 = _mm_mul_pd(a22, b2);		
		c2 = _mm_add_pd(c2, a22);			
		_mm_store_pd(C, c1);
		_mm_store_pd(C+2, c2);	
}

inline mmult_4x4(const double* A, const double* B, double* C) {
  int k;
  __m128d a1, a2, a3, a4, b1a, b2a, b3a, b4a, c1, b1b, b2b, b3b, b4b, c2;

      for (k = 0; k < 4; k++) {
	      a1 = _mm_load1_pd(A);
	      a2 = _mm_load1_pd(A + 1);
	      a3 = _mm_load1_pd(A + 2);
	      a4 = _mm_load1_pd(A + 3);
	      
        b1a = _mm_load_pd(B);
        b2a = _mm_load_pd(B + 4);
        b3a = _mm_load_pd(B + 8);
        b4a = _mm_load_pd(B + 12);
        b1b = _mm_load_pd(B + 2);
        b2b = _mm_load_pd(B + 6);
        b3b = _mm_load_pd(B + 10);
        b4b = _mm_load_pd(B + 14);
        c1 = _mm_load_pd(C);
        c2 = _mm_load_pd(C + 2);

        b1a = _mm_mul_pd(a1, b1a);
        c1 = _mm_add_pd(b1a, c1);
        b2a = _mm_mul_pd(a2, b2a);
        c1 = _mm_add_pd(b2a, c1);
        b3a = _mm_mul_pd(a3, b3a);
        c1 = _mm_add_pd(b3a, c1);
        b4a = _mm_mul_pd(a4, b4a);
        c1 = _mm_add_pd(b4a, c1);

        b1b = _mm_mul_pd(a1, b1b);
        c2 = _mm_add_pd(b1b, c2);
        b2b = _mm_mul_pd(a2, b2b);
        c2 = _mm_add_pd(b2b, c2);
        b3b = _mm_mul_pd(a3, b3b);
        c2 = _mm_add_pd(b3b, c2);
        b4b = _mm_mul_pd(a4, b4b);
        c2 = _mm_add_pd(b4b, c2);

        _mm_store_pd(C, c1);
        _mm_store_pd(C + 2, c2);
        C += 4;
        A += 4;
    }
}

inline mmult_8x8(const double* A, const double* B, double* C) {
  int i, j, k;
  __m128d a1, a2, a3, a4, b1a, b2a, b3a, b4a, c1, b1b, b2b, b3b, b4b, c2;

      for (i = 0; i < 8; i++) {  //move A vertical
      	for (j = 0; j < 2; j++){  //move A horizontal
      	  const double* B0 = B + j * 32;
      	        double* C0 = C;
      	  a1 = _mm_load1_pd(A);
	      a2 = _mm_load1_pd(A + 1);
	      a3 = _mm_load1_pd(A + 2);
	      a4 = _mm_load1_pd(A + 3);
	      
      	  for (k = 0; k < 2; k++){ //move B horizontal
        b1a = _mm_load_pd(B0);
        b2a = _mm_load_pd(B0 + 8);
        b3a = _mm_load_pd(B0 + 16);
        b4a = _mm_load_pd(B0 + 24);
        B0 += 2;
        b1b = _mm_load_pd(B0);
        b2b = _mm_load_pd(B0 + 8);
        b3b = _mm_load_pd(B0 + 16);
        b4b = _mm_load_pd(B0 + 24);
        B0 += 2;
        c1 = _mm_load_pd(C0);
        c2 = _mm_load_pd(C0 + 2);

        b1a = _mm_mul_pd(a1, b1a);
        c1 = _mm_add_pd(b1a, c1);
        b2a = _mm_mul_pd(a2, b2a);
        c1 = _mm_add_pd(b2a, c1);
        b3a = _mm_mul_pd(a3, b3a);
        c1 = _mm_add_pd(b3a, c1);
        b4a = _mm_mul_pd(a4, b4a);
        c1 = _mm_add_pd(b4a, c1);

        b1b = _mm_mul_pd(a1, b1b);
        c2 = _mm_add_pd(b1b, c2);
        b2b = _mm_mul_pd(a2, b2b);
        c2 = _mm_add_pd(b2b, c2);
        b3b = _mm_mul_pd(a3, b3b);
        c2 = _mm_add_pd(b3b, c2);
        b4b = _mm_mul_pd(a4, b4b);
        c2 = _mm_add_pd(b4b, c2);

        _mm_store_pd(C0, c1);
        _mm_store_pd(C0 + 2, c2);		   
        C0 +=4;  	      	
     }
       A +=4;
    }
	C += 8;
   }
}

inline mmult_16x16(const double* A, const double* B, double* C) {
  int i, j, k;
  __m128d a1, a2, a3, a4, b1a, b2a, b3a, b4a, c1, b1b, b2b, b3b, b4b, c2;
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 4; j++) {
      double* C0 = C;
      const double* B0 = B + j * 64;
      a1 = _mm_load1_pd(A);
      a2 = _mm_load1_pd(A + 1);
      a3 = _mm_load1_pd(A + 2);
      a4 = _mm_load1_pd(A + 3);
      for (k = 0; k < 4; k++) {
        b1a = _mm_load_pd(B0);
        b2a = _mm_load_pd(B0 + 16);
        b3a = _mm_load_pd(B0 + 32);
        b4a = _mm_load_pd(B0 + 48);
        B0 += 2;
        b1b = _mm_load_pd(B0);
        b2b = _mm_load_pd(B0 + 16);
        b3b = _mm_load_pd(B0 + 32);
        b4b = _mm_load_pd(B0 + 48);
        B0 += 2;
        c1 = _mm_load_pd(C0);
        c2 = _mm_load_pd(C0 + 2);

        b1a = _mm_mul_pd(a1, b1a);
        c1 = _mm_add_pd(b1a, c1);
        b2a = _mm_mul_pd(a2, b2a);
        c1 = _mm_add_pd(b2a, c1);
        b3a = _mm_mul_pd(a3, b3a);
        c1 = _mm_add_pd(b3a, c1);
        b4a = _mm_mul_pd(a4, b4a);
        c1 = _mm_add_pd(b4a, c1);

        b1b = _mm_mul_pd(a1, b1b);
        c2 = _mm_add_pd(b1b, c2);
        b2b = _mm_mul_pd(a2, b2b);
        c2 = _mm_add_pd(b2b, c2);
        b3b = _mm_mul_pd(a3, b3b);
        c2 = _mm_add_pd(b3b, c2);
        b4b = _mm_mul_pd(a4, b4b);
        c2 = _mm_add_pd(b4b, c2);

        _mm_store_pd(C0, c1);
        _mm_store_pd(C0 + 2, c2);
        C0 += 4;
      }
      A += 4;
    }
    C += 16;
  }
}

inline mmult_NxN(int N, const double* A, const double* B, double* C) {
  int i, j, k;
  __m128d a1, a2, a3, a4, b1a, b2a, b3a, b4a, c1, b1b, b2b, b3b, b4b, c2;

      for (i = 0; i < N; i++) {  //move A vertical
      	for (j = 0; j < N/4; j++){  //move A horizontal
      	  const double* B0 = B + j * 4 * N;
      	        double* C0 = C;
      	  a1 = _mm_load1_pd(A);
	      a2 = _mm_load1_pd(A + 1);
	      a3 = _mm_load1_pd(A + 2);
	      a4 = _mm_load1_pd(A + 3);
	      
      	  for (k = 0; k < N/4; k++){ //move B horizontal
        b1a = _mm_load_pd(B0);
        b2a = _mm_load_pd(B0 + N);
        b3a = _mm_load_pd(B0 + 2*N);
        b4a = _mm_load_pd(B0 + 3*N);
        B0 += 2;
        b1b = _mm_load_pd(B0);
        b2b = _mm_load_pd(B0 + N  );
        b3b = _mm_load_pd(B0 + 2*N);
        b4b = _mm_load_pd(B0 + 3*N);
        B0 += 2;
        c1 = _mm_load_pd(C0);
        c2 = _mm_load_pd(C0 + 2);

        b1a = _mm_mul_pd(a1, b1a);
        c1 = _mm_add_pd(b1a, c1);
        b2a = _mm_mul_pd(a2, b2a);
        c1 = _mm_add_pd(b2a, c1);
        b3a = _mm_mul_pd(a3, b3a);
        c1 = _mm_add_pd(b3a, c1);
        b4a = _mm_mul_pd(a4, b4a);
        c1 = _mm_add_pd(b4a, c1);

        b1b = _mm_mul_pd(a1, b1b);
        c2 = _mm_add_pd(b1b, c2);
        b2b = _mm_mul_pd(a2, b2b);
        c2 = _mm_add_pd(b2b, c2);
        b3b = _mm_mul_pd(a3, b3b);
        c2 = _mm_add_pd(b3b, c2);
        b4b = _mm_mul_pd(a4, b4b);
        c2 = _mm_add_pd(b4b, c2);

        _mm_store_pd(C0, c1);
        _mm_store_pd(C0 + 2, c2);		   
        C0 += 4;  	      	
     }
       A +=4;
    }
	C += N;
   }
}

inline mmult_64x64(const double* A, const double* B, double* C) {
  int i, j, k;
  __m128d a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, c1, b5, b6, b7, b8;

      for (i = 0; i < 64; i++) {  //move A vertical
      	for (j = 0; j < 8; j++){  //move A horizontal
      	  const double* B0 = B + j * 64 * 8;
      	        double* C0 = C;
      	  a1 = _mm_load1_pd(A);
	      a2 = _mm_load1_pd(A + 1);
	      a3 = _mm_load1_pd(A + 2);
	      a4 = _mm_load1_pd(A + 3);
	      a5 = _mm_load1_pd(A + 4);
	      a6 = _mm_load1_pd(A + 5);
	      a7 = _mm_load1_pd(A + 6);
	      a8 = _mm_load1_pd(A + 7);
	      
      	  for (k = 0; k < 8; k++){ //move B horizontal
        b1 = _mm_load_pd(B0);
        b2 = _mm_load_pd(B0 + 64);
        b3 = _mm_load_pd(B0 + 2*64);
        b4 = _mm_load_pd(B0 + 3*64);
        b5 = _mm_load_pd(B0 + 4*64);
        b6 = _mm_load_pd(B0 + 5*64);
        b7 = _mm_load_pd(B0 + 6*64);
        b8 = _mm_load_pd(B0 + 7*64);
        c1 = _mm_load_pd(C0);
        b1 = _mm_mul_pd(a1, b1);
        c1 = _mm_add_pd(b1, c1);
        b2 = _mm_mul_pd(a2, b2);
        c1 = _mm_add_pd(b2, c1);
        b3 = _mm_mul_pd(a3, b3);
        c1 = _mm_add_pd(b3, c1);
        b4 = _mm_mul_pd(a4, b4);
        c1 = _mm_add_pd(b4, c1);
        b5 = _mm_mul_pd(a5, b5);
        c1 = _mm_add_pd(b5, c1);
        b6 = _mm_mul_pd(a6, b6);
        c1 = _mm_add_pd(b6, c1);
        b7 = _mm_mul_pd(a7, b7);
        c1 = _mm_add_pd(b7, c1);
        b8 = _mm_mul_pd(a8, b8);
        c1 = _mm_add_pd(b8, c1);
        _mm_store_pd(C0, c1);      
        B0 += 2;
        
 		b1 = _mm_load_pd(B0);
        b2 = _mm_load_pd(B0 + 64);
        b3 = _mm_load_pd(B0 + 2*64);
        b4 = _mm_load_pd(B0 + 3*64);
        b5 = _mm_load_pd(B0 + 4*64);
        b6 = _mm_load_pd(B0 + 5*64);
        b7 = _mm_load_pd(B0 + 6*64);
        b8 = _mm_load_pd(B0 + 7*64);
        c1 = _mm_load_pd(C0 + 2);
        b1 = _mm_mul_pd(a1, b1);
        c1 = _mm_add_pd(b1, c1);
        b2 = _mm_mul_pd(a2, b2);
        c1 = _mm_add_pd(b2, c1);
        b3 = _mm_mul_pd(a3, b3);
        c1 = _mm_add_pd(b3, c1);
        b4 = _mm_mul_pd(a4, b4);
        c1 = _mm_add_pd(b4, c1);
        b5 = _mm_mul_pd(a5, b5);
        c1 = _mm_add_pd(b5, c1);
        b6 = _mm_mul_pd(a6, b6);
        c1 = _mm_add_pd(b6, c1);
        b7 = _mm_mul_pd(a7, b7);
        c1 = _mm_add_pd(b7, c1);
        b8 = _mm_mul_pd(a8, b8);
        c1 = _mm_add_pd(b8, c1);
        _mm_store_pd(C0 + 2, c1);
        B0 += 2;

 		b1 = _mm_load_pd(B0);
        b2 = _mm_load_pd(B0 + 64);
        b3 = _mm_load_pd(B0 + 2*64);
        b4 = _mm_load_pd(B0 + 3*64);
        b5 = _mm_load_pd(B0 + 4*64);
        b6 = _mm_load_pd(B0 + 5*64);
        b7 = _mm_load_pd(B0 + 6*64);
        b8 = _mm_load_pd(B0 + 7*64);
        c1 = _mm_load_pd(C0 + 4);
        b1 = _mm_mul_pd(a1, b1);
        c1 = _mm_add_pd(b1, c1);
        b2 = _mm_mul_pd(a2, b2);
        c1 = _mm_add_pd(b2, c1);
        b3 = _mm_mul_pd(a3, b3);
        c1 = _mm_add_pd(b3, c1);
        b4 = _mm_mul_pd(a4, b4);
        c1 = _mm_add_pd(b4, c1);
        b5 = _mm_mul_pd(a5, b5);
        c1 = _mm_add_pd(b5, c1);
        b6 = _mm_mul_pd(a6, b6);
        c1 = _mm_add_pd(b6, c1);
        b7 = _mm_mul_pd(a7, b7);
        c1 = _mm_add_pd(b7, c1);
        b8 = _mm_mul_pd(a8, b8);
        c1 = _mm_add_pd(b8, c1);
        _mm_store_pd(C0 + 4, c1);
        B0 += 2;
        
 		b1 = _mm_load_pd(B0);
        b2 = _mm_load_pd(B0 + 64);
        b3 = _mm_load_pd(B0 + 2*64);
        b4 = _mm_load_pd(B0 + 3*64);
        b5 = _mm_load_pd(B0 + 4*64);
        b6 = _mm_load_pd(B0 + 5*64);
        b7 = _mm_load_pd(B0 + 6*64);
        b8 = _mm_load_pd(B0 + 7*64);
        c1 = _mm_load_pd(C0 + 6);
        b1 = _mm_mul_pd(a1, b1);
        c1 = _mm_add_pd(b1, c1);
        b2 = _mm_mul_pd(a2, b2);
        c1 = _mm_add_pd(b2, c1);
        b3 = _mm_mul_pd(a3, b3);
        c1 = _mm_add_pd(b3, c1);
        b4 = _mm_mul_pd(a4, b4);
        c1 = _mm_add_pd(b4, c1);
        b5 = _mm_mul_pd(a5, b5);
        c1 = _mm_add_pd(b5, c1);
        b6 = _mm_mul_pd(a6, b6);
        c1 = _mm_add_pd(b6, c1);
        b7 = _mm_mul_pd(a7, b7);
        c1 = _mm_add_pd(b7, c1);
        b8 = _mm_mul_pd(a8, b8);
        c1 = _mm_add_pd(b8, c1);
        _mm_store_pd(C0 + 6, c1);
        B0 += 2;   
        C0 += 8;  	      	
     }
       A +=8;
    }
	C += 64;
   }
}

inline madd(int N, 
          int Xpitch, const double* X, 
          int Ypitch, const double* Y,
          int Spitch, double* S) {
  int i, j;
  __m128d x, y, z;
   for ( i = 0; i != N; i++){
    for (j = 0; j != N; j+=2){

   	  x = _mm_load_pd(X + i*Xpitch+j);
   	  y = _mm_load_pd(Y+i*Ypitch+j);
   	  z = _mm_add_pd(x, y);
   	  _mm_store_pd(S+i*Spitch+j, z);
    }
   }

}

inline msub(int N, 
          int Xpitch, const double* X, 
          int Ypitch, const double* Y,
          int Spitch, double* S) {
  int i, j;
  __m128d x, y, z;
   for ( i = 0; i != N; i++){
    for (j = 0; j != N; j+=2){

   	  x = _mm_load_pd(X + i*Xpitch+j);
   	  y = _mm_load_pd(Y+i*Ypitch+j);
   	  z = _mm_sub_pd(x, y);
   	  _mm_store_pd(S+i*Spitch+j, z);
    }
   }
}  

inline mmult_noadd(int N, int Apitch, const double* A, 
           int Bpitch, const double* B,
           int Cpitch, double* C) {
  int i, j, k;
  __m128d a1, a2, a3, a4, b1a, b2a, b3a, b4a, c1, b1b, b2b, b3b, b4b, c2;
  for (i = 0; i < N; i++) {
      double* A0 = A;
    for (j = 0; j < N/4; j++) {
      double* C0 = C;
      const double* B0 = B + j * 4 * Bpitch;
      a1 = _mm_load1_pd(A0);
      a2 = _mm_load1_pd(A0 + 1);
      a3 = _mm_load1_pd(A0 + 2);
      a4 = _mm_load1_pd(A0 + 3);
      for (k = 0; k < N/4; k++) {
        b1a = _mm_load_pd(B0);
        b2a = _mm_load_pd(B0 + Bpitch);
        b3a = _mm_load_pd(B0 + 2*Bpitch);
        b4a = _mm_load_pd(B0 + 3*Bpitch);
        B0 += 2;
        b1b = _mm_load_pd(B0);
        b2b = _mm_load_pd(B0 + Bpitch);
        b3b = _mm_load_pd(B0 + 2*Bpitch);
        b4b = _mm_load_pd(B0 + 3*Bpitch);
        B0 += 2;
        if (j==0){
 			c1 = _mm_setzero_pd();
		 	c2 = _mm_setzero_pd();
        }
        else {
        	c1 = _mm_load_pd(C0);
        	c2 = _mm_load_pd(C0 + 2);	
        }
        b1a = _mm_mul_pd(a1, b1a);
        c1 = _mm_add_pd(b1a, c1);
        b2a = _mm_mul_pd(a2, b2a);
        c1 = _mm_add_pd(b2a, c1);
        b3a = _mm_mul_pd(a3, b3a);
        c1 = _mm_add_pd(b3a, c1);
        b4a = _mm_mul_pd(a4, b4a);
        c1 = _mm_add_pd(b4a, c1);

        b1b = _mm_mul_pd(a1, b1b);
        c2 = _mm_add_pd(b1b, c2);
        b2b = _mm_mul_pd(a2, b2b);
        c2 = _mm_add_pd(b2b, c2);
        b3b = _mm_mul_pd(a3, b3b);
        c2 = _mm_add_pd(b3b, c2);
        b4b = _mm_mul_pd(a4, b4b);
        c2 = _mm_add_pd(b4b, c2);

        _mm_store_pd(C0, c1);
        _mm_store_pd(C0 + 2, c2);
        C0 += 4;
      }
      A0 += 4;
    }
    A += Apitch;
    C += Cpitch;
  }
  }
  
// Volker Strassen algorithm for matrix multiplication.

void mmult_fast(int N, 
                int Xpitch, const double* X, 
                int Ypitch, const double* Y,
                int Zpitch, double* Z) {
  if (N == 32) {
	mmult_noadd(N, Xpitch, X, Ypitch, Y, Zpitch, Z);
    return;
  }

  const int n = N/2;      // size of sub-matrices

  const double * A = X;    // A-D matrices embedded in X
  const double * B = X + n;
  const double * C = X + n*Xpitch;
  const double * D = C + n;

  const double * E = Y;    // E-H matrices embeded in Y
  const double * F = Y + n;
  const double * G = Y + n*Ypitch;
  const double * H = G + n;

  double * P[7];   // allocate temp matrices off heap
  const int sz = n*n*sizeof(double);
  int i;
  for (i = 0; i != 7; i++)
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

  for (i = 0; i != 7; i++){
  	    free(P[i]);
  }
}

void matmul(int N, const double* A, const double* B, double* C) {
    if (N == 2){
    mmult_2x2(A, B, C);	
    }
    else if (N == 4){
    mmult_4x4(A, B, C);	
    }
    else if (N == 8){
    mmult_8x8(A, B, C);
    }
    else if (N == 16){
    mmult_16x16(A, B, C);
    }

    else if (N == 32){
   	mmult_NxN(N, A, B, C);
    }
	else if (N == 64){
   	mmult_64x64(A, B, C);
    }
    else{
   	double *D = (double*) malloc(N*N*sizeof(double));
	mmult_fast(N, N, A, N, B, N, D);
	madd(N, N, C, N, D, N, C);
    }

}

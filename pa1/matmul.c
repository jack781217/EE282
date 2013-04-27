// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

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
	#1: added blocking
*/
void matmul(int N, const double* A, const double* B, double* C) {
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

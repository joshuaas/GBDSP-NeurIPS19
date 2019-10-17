#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Variables */
    int n,e,n1,n2,edge,
			nNodes,nEdges,
			*nNei,
			*edgeEnds;
	double *V,*E;
	
	/* Inputs */
	edgeEnds = (int*)mxGetPr(prhs[0]);
	nNodes = (int)mxGetScalar(prhs[1]);
    nEdges = mxGetDimensions(prhs[0])[0];
    
	/* Allocate memory */
	nNei = mxCalloc(nNodes,sizeof(int));
	
	/* Outputs */
	plhs[0] = mxCreateDoubleMatrix(nNodes+1,1,mxREAL);
	plhs[1] = mxCreateDoubleMatrix(2*nEdges,1,mxREAL);
	V = mxGetPr(plhs[0]);
	E = mxGetPr(plhs[1]);
	
	/* Count number of neighbors for each node */
	for(e=0;e<nEdges;e++) {
		n1 = edgeEnds[e];
		n2 = edgeEnds[e+nEdges];
		nNei[n1]++;
		nNei[n2]++;
	}
    
    /* Make V structure */
    edge = 0;
    for(n=0;n<nNodes;n++) {
        V[n] = (double)edge+1;
        edge = edge + nNei[n];
    }
    V[nNodes] = (double)edge+1;

    /* Reset number of neighbors (info now contained in V) */
    for(n=0;n<nNodes;n++) {
        nNei[n] = 0;
    }
    
    /* Make E structure */
    for(e=0;e<nEdges;e++) {
        n1 = edgeEnds[e];
        n2 = edgeEnds[e+nEdges];
        E[(int)V[n1]-1+nNei[n1]++] = e+1;
        E[(int)V[n2]-1+nNei[n2]++] = e+1;
    }
    
	
	/* Free all memory */
	mxFree(nNei);
}

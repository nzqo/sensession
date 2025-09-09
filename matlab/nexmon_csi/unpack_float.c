#include "mex.h"
#include "matrix.h"
#include <stdint.h>

#define k_tof_unpack_sgn_mask (1<<31)

void unpack_float_acphy(int nbits, int autoscale, int shft, 
                       int fmt, int nman, int nexp, int nfft, 
                       uint32_T *H, int32_T *Hout, int autoshift)
{
        int e_p, maxbit, e, i, pwr_shft = 0, e_zero, sgn;
        int n_out, e_shift;
        int8_t He[8192]; // worst case, debug mode
        int32_t vi, vq, *pOut;
        uint32_t x, iq_mask, e_mask, sgnr_mask, sgni_mask;

        if (fmt)
                iq_mask = (1<<(nman-1))- 1;
        else
                iq_mask = (1 << nman) - 1;

        e_mask = (1<<nexp)-1;
        e_p = (1<<(nexp-1));

        if (fmt) {
                sgnr_mask = (1 << (nexp + 2*nman - 1));
                sgni_mask = (sgnr_mask >> nman);
        } else {
                sgnr_mask = (1 << 2 * nman);
                sgni_mask = (sgnr_mask << 1);
        }
        e_zero = -nman;
        pOut = (int32_t*)Hout;
        n_out = (nfft << 1);
        e_shift = 1;

        maxbit = -e_p;
        for (i = 0; i < nfft; i++) {
                if (fmt) {
                        vi = (int32_t)((H[i] >> (nexp + nman)) & iq_mask);
                        vq = (int32_t)((H[i] >> nexp) & iq_mask);
                        e =   (int)(H[i] & e_mask);
                } else {
                        vi = (int32_t)(H[i] & iq_mask);
                        vq = (int32_t)((H[i] >> nman) & iq_mask);
                        e = (int32_t)((H[i] >> (2 * nman + 2)) & e_mask);
                }
                if (e >= e_p)
                        e -= (e_p << 1);
                He[i] = (int8_t)e;
                x = (uint32_t)vi | (uint32_t)vq;
                if (autoscale && x) {
                        uint32_t m = 0xffff0000, b = 0xffff;
                        int s = 16;
                        while (s > 0) {
                                if (x & m) {
                                        e += s;
                                        x >>= s;
                                }
                                s >>= 1;
                                m = (m >> s) & b;
                                b >>= s;
                        }
                        if (e > maxbit)
                                maxbit = e;
                }
                if (H[i] & sgnr_mask)
                        vi |= k_tof_unpack_sgn_mask;
                if (H[i] & sgni_mask)
                        vq |= k_tof_unpack_sgn_mask;
                Hout[i<<1] = vi;
                Hout[(i<<1)+1] = vq;
        }
        if (autoshift)
                shft = nbits - maxbit;
        else
                shft = 16;
        for (i = 0; i < n_out; i++) {
                e = He[(i >> e_shift)] + shft;
                vi = *pOut;
                sgn = 1;
                if (vi & k_tof_unpack_sgn_mask) {
                        sgn = -1;
                        vi &= ~k_tof_unpack_sgn_mask;
                }
                if (e < e_zero) {
                        vi = 0;
                } else if (e < 0) {
                        e = -e;
                        vi = (vi >> e);
                } else {
                        vi = (vi << e);
                }
                *pOut++ = (int32_t)sgn*vi;
        }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int *nfftp, *formatp;
    uint32_T *H;
    int32_T *Hout;
    int *autoshiftp;
    mwSize Hsize;
    if(nrhs != 4) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:nrhs",
                          "Four inputs expected: unpack_float(int32 format, int32 autoshift, int32 nfft, uint32 H[](nfftx1))");
    }
    if(!mxIsInt32(prhs[0])) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:formatNotInt32",
                          "Input format must be Int32.");
    }
    if(!mxIsInt32(prhs[1])) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:formatNotInt32",
                          "Input format must be Int32.");
    }
    if(!mxIsInt32(prhs[2])) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:nfftNotInt32",
                          "Input nfft must be Int32.");
    }
    if(!mxIsUint32(prhs[3])) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:HNotInt32",
                          "Input H must be Uint32.");
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:nlhs",
                          "One output required.");
    }

    formatp = (int*)mxGetData(prhs[0]);
    autoshiftp = (int*)mxGetData(prhs[1]);
    nfftp = (int*)mxGetData(prhs[2]);
    Hsize = mxGetNumberOfElements(prhs[3]);
    if(Hsize < *nfftp) {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:Hsize",
                          "Length of H must be at least nfft.");
    }
    H = (uint32_T*)mxGetData(prhs[3]);
    plhs[0] = mxCreateNumericMatrix((*nfftp)<<1, 1, mxINT32_CLASS, mxREAL);
    Hout = (int32_T*)mxGetPr(plhs[0]);
    if (*formatp == 0) {
        unpack_float_acphy(10, 1, 0, 1, 9, 5, *nfftp, H, Hout, *autoshiftp);
    } else if (*formatp == 1) {
        unpack_float_acphy(10, 1, 0, 1, 12, 6, *nfftp, H, Hout, *autoshiftp);
    } else {
        mexErrMsgIdAndTxt("NexmonCSI:unpack_float:format",
                          "format can only be 0 or 1.");
    }
}

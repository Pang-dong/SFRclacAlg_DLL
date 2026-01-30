#include "SfrExports.h"
#include "SFRClacAlg.h"

extern "C" __declspec(dllexport)
int __cdecl EdgeSFR_HB(
    unsigned char* img,
    int imgW,
    int imgH,
    double cyclepixel,
    double* sfrVal)
{
    if (!img || !sfrVal) return -1;
    double v = 0.0;
    int ret = CSFRClacAlg::EdgeSFR_HB(img, imgW, imgH, cyclepixel, v);
    *sfrVal = v;
    return ret;
}
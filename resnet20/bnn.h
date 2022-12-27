#ifndef BNN_H
#define BNN_H

#include "typedefs.h"
#include "dimension_def.h"

void FracNet_T(
	float image[3][32][32],
	float ctrl_tl,
	float &loss,
	float error[10]
);

#endif


#ifndef TYPEDEFS
#define TYPEDEFS

#include <cstddef>
#include <ap_int.h>
#include <ap_fixed.h>

#define SW_TEST
#define LAYER_TEST
#define True 1
#define False 0

/* 
 *bm(2,5)
 */

/*
#define exp 2
#define man 5
#define tile 1024	// 32*32
#define emax 1	// 3-2
#define emin -2
#define max_number 2*(2 - 1/32)	// 2**(self.emax)*(2-2**(-self.man))
*/

#ifdef SW_TEST
	typedef float FIX_32_4;		//fix point
	typedef float FIX_32_25;	//fix point
	typedef float FIX_FM;		//fix point for feature map
	typedef float FIX_FM_acc;	//fix point for feature map
	typedef float FIX_FM_last;
	typedef float FIX_WT;		//fix point for weights
	typedef float FIX_32_16;
	typedef float FIX_32_10;
	typedef float FIX_32_12;
	typedef float FIX_16_6;
	typedef float FIX_16_5;
	typedef float FIX_16_4;
	typedef float FIX_16_10;

#else
	typedef ap_fixed<16, 9, AP_RND, AP_SAT> FIX_FM_acc;	//fix point for accumulation (16, 8) (20,9 works)
	typedef ap_fixed<12, 4, AP_RND, AP_SAT> FIX_WT;	//fix point for batchnorm weights (16, 4 works)

	typedef ap_fixed<32,12, AP_RND, AP_SAT> FIX_32_12;
	typedef ap_fixed<32,10, AP_RND, AP_SAT> FIX_32_10;

#endif

	typedef ap_uint<1> uint1;
	typedef ap_uint<2> uint2;
	typedef ap_uint<3> uint3;
	typedef ap_uint<4> uint4;
	typedef ap_uint<5> uint5;
	typedef ap_uint<6> uint6;
	typedef ap_uint<7> uint7;
	typedef ap_uint<8> uint8;
	typedef ap_uint<10> uint10;
	typedef ap_uint<12> uint12;
	typedef ap_uint<16> uint16;
	typedef ap_uint<32> uint32;
	typedef ap_uint<64> uint64;
	typedef ap_uint<128> uint128;
	typedef ap_uint<256> uint256;
	typedef ap_uint<512> uint512;

	typedef ap_int<1> int1;
	typedef ap_int<2> int2;
	typedef ap_int<3> int3;
	typedef ap_int<4> int4;
	typedef ap_int<5> int5;
	typedef ap_int<6> int6;
	typedef ap_int<7> int7;
	typedef ap_int<8> int8;
	typedef ap_int<9> int9;
	typedef ap_int<10> int10;
	typedef ap_int<12> int12;
	typedef ap_int<14> int14;
	typedef ap_int<16> int16;
	typedef ap_int<20> int20;
	typedef ap_int<32> int32;
	typedef ap_int<64> int64;
	typedef ap_int<128> int128;
	typedef ap_int<256> int256;
	typedef ap_int<512> int512;

	// bm(2,5): (1-bit sgn + 5-bit man) + 2-bit exp
	// typedef ap_int<8> int8;

	// mantissa (for normals)
	typedef ap_fixed<7, 1, AP_RND, AP_WRAP> FIX_1_5;
	typedef ap_fixed<8, 2, AP_RND, AP_WRAP> FIX_2_5;
	typedef ap_fixed<9, 3, AP_RND, AP_WRAP> FIX_3_5;
	typedef ap_fixed<10, 4, AP_RND, AP_WRAP> FIX_4_5;
	typedef ap_fixed<13, 5, AP_RND, AP_WRAP> FIX_7_5;

	// exponent
	// typedef ap_int<2> int2;

#endif

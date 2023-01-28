#ifndef XBR_UTILS
#define XBR_UTILS

#include "img_utils.h"
#include "xBR_interpolations.h"


__device__ unsigned int d(const PixelYUV a, const PixelYUV b) {
	return 48 * abs(a.Y - b.Y) + 7 * abs(a.U - b.U) + 6 * abs(a.V - b.V);
}

__device__ int YUV_equals(const PixelYUV a, const PixelYUV b) {
	return d(a, b) <= 600;
}

/**
 * Naming conventions
 *
 * Pixels:
 *    A1 B1 C1
 * A0 A  B  C  C4
 * D0 D  E  F  F4
 * G0 G  H  I  I4
 *    G5 H5 I5
 *
 * wd = weighted distance
 *
 * directions
 * BR = bottom right
 * BL = bottom left
 * TL = top left
 * TR = top right
 *
 * O|P = orthogonal|parallel directions, see images
 *
 * ex: wdBRO = weighted distance, bottom right direction, orthogonal
 **/
__device__ void expand_pixel_tiling(const unsigned int cols, const unsigned int tileCols, PixelRGB *inputRGB, PixelYUV *inputYUV, PixelRGB *output, const unsigned int scaleFactor, const unsigned int t_row, const unsigned int t_col, const unsigned int o_row, const unsigned int o_col) {
	bool edgeBR;
	bool edgeBL;
	bool edgeTL;
	bool edgeTR;

	// edge detection
	PixelYUV A = inputYUV[(t_row-1) * tileCols + (t_col-1)];
	PixelYUV B = inputYUV[(t_row-1) * tileCols + t_col];
	PixelYUV C = inputYUV[(t_row-1) * tileCols + (t_col+1)];
	PixelYUV D = inputYUV[t_row * tileCols + (t_col-1)];
	PixelYUV E = inputYUV[t_row * tileCols + t_col];
	PixelYUV F = inputYUV[t_row * tileCols + (t_col+1)];
	PixelYUV G = inputYUV[(t_row+1) * tileCols + (t_col-1)];
	PixelYUV H = inputYUV[(t_row+1) * tileCols + t_col];
	PixelYUV I = inputYUV[(t_row+1) * tileCols + (t_col+1)];

	PixelYUV A1 = inputYUV[(t_row-2) * tileCols + (t_col-1)];
	PixelYUV B1 = inputYUV[(t_row-2) * tileCols + t_col];
	PixelYUV C1 = inputYUV[(t_row-2) * tileCols + (t_col+1)];
	PixelYUV A0 = inputYUV[(t_row-1) * tileCols + (t_col-2)];
	PixelYUV C4 = inputYUV[(t_row-1) * tileCols + (t_col+2)];
	PixelYUV D0 = inputYUV[t_row * tileCols + (t_col-2)];
	PixelYUV F4 = inputYUV[t_row * tileCols + (t_col+2)];
	PixelYUV G0 = inputYUV[(t_row+1) * tileCols + (t_col-2)];
	PixelYUV I4 = inputYUV[(t_row+1) * tileCols + (t_col+2)];
	PixelYUV G5 = inputYUV[(t_row+2) * tileCols + (t_col-1)];
	PixelYUV H5 = inputYUV[(t_row+2) * tileCols + t_col];
	PixelYUV I5 = inputYUV[(t_row+2) * tileCols + (t_col+1)];

	unsigned int wdBRO = d(C, E) + d(E, G) + d(F4, I) + d(I, H5) + 4*d(F, H);
	unsigned int wdBRP = d(B, F) + d(F, I4) + d(D, H) + d(H, I5) + 4*d(E, I);
	edgeBR = wdBRO < wdBRP;

	unsigned int wdBLO = d(A, E) + d(E, I) + d(D0, G) + d(G, H5) + 4*d(D, H);
	unsigned int wdBLP = d(B, D) + d(D, G0) + d(F, H) + d(H, G5) + 4*d(E, G);
	edgeBL = wdBLO < wdBLP;

	unsigned int wdTLO = d(C, E) + d(E, G) + d(B1, A) + d(A, D0) + 4*d(B, D);
	unsigned int wdTLP = d(A1, B) + d(B, F) + d(A0, D) + d(D, H) + 4*d(A, E);
	edgeTL = wdTLO < wdTLP;

	unsigned int wdTRO = d(A, E) + d(E, I) + d(B1, C) + d(C, F4) + 4*d(B, F);
	unsigned int wdTRP = d(C1, B) + d(B, D) + d(C4, F) + d(F, H) + 4*d(C, E);
	edgeTR = wdTRO < wdTRP;


	// interpolation
	unsigned int out_cols = cols * scaleFactor;
	PixelRGB *out = &output[o_row * out_cols * scaleFactor + o_col * scaleFactor];

	PixelRGB E_rgb = inputRGB[t_row * tileCols + t_col];
	for (int r = 0; r < scaleFactor; r++) {
		for (int c = 0; c < scaleFactor; c++) {
			out[r * out_cols + c] = E_rgb;
		}
	}

	if (edgeBR) {
		PixelRGB newColor = d(E, F) <= d(E, H) ? inputRGB[t_row * tileCols + (t_col+1)] : inputRGB[(t_row+1) * tileCols + t_col];

		if (YUV_equals(F, G) && YUV_equals(C, H) && !edgeBL && !edgeTR) {
			BR_INT2_BR(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(F, G) && !edgeBL)
			BR_INT2_B(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(C, H) && !edgeTR)
			BR_INT2_R(scaleFactor, out, out_cols, newColor);
		else
			BR_INT1(scaleFactor, out, out_cols, newColor);
	}

	if (edgeBL) {
		PixelRGB newColor = d(D, E) <= d(E, H) ? inputRGB[t_row * tileCols + (t_col-1)] : inputRGB[(t_row+1) * tileCols + t_col];

		if (YUV_equals(D, I) && YUV_equals(A, H) && !edgeBR && !edgeTL) {
			BL_INT2_BL(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(D, I) && !edgeBR)
			BL_INT2_B(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(A, H) && !edgeTL)
			BL_INT2_L(scaleFactor, out, out_cols, newColor);
		else
			BL_INT1(scaleFactor, out, out_cols, newColor);
	}

	if (edgeTL) {
		PixelRGB newColor = d(B, E) <= d(D, E) ? inputRGB[(t_row-1) * tileCols + t_col] : inputRGB[t_row * tileCols + (t_col-1)];

		if (YUV_equals(C, D) && YUV_equals(B, G) && !edgeTR && !edgeBL) {
			TL_INT2_TL(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(C, D) && !edgeTR)
			TL_INT2_T(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(B, G) && !edgeBL)
			TL_INT2_L(scaleFactor, out, out_cols, newColor);
		else
			TL_INT1(scaleFactor, out, out_cols, newColor);
	}

	if (edgeTR) {
		PixelRGB newColor = d(B, E) <= d(E, F) ? inputRGB[(t_row-1) * tileCols + t_col] : inputRGB[t_row * tileCols + (t_col+1)];

		if (YUV_equals(A, F) && YUV_equals(B, I) && !edgeTL && !edgeBR) {
			TR_INT2_TR(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(A, F) && !edgeTL)
			TR_INT2_T(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(B, I) && !edgeBR)
			TR_INT2_R(scaleFactor, out, out_cols, newColor);
		else
			TR_INT1(scaleFactor, out, out_cols, newColor);
	}

}

#endif

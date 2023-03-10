#ifndef XBR_UTILS
#define XBR_UTILS

#include "img_utils.h"
#include "xBR_interpolations.h"


unsigned int d(const PixelYUV a, const PixelYUV b) {
	return 48 * abs(a.Y - b.Y) + 7 * abs(a.U - b.U) + 6 * abs(a.V - b.V);
}

int YUV_equals(PixelYUV a, PixelYUV b) {
	return d(a, b) <= 600;
}

PixelYUV get_YUV(unsigned int rows, unsigned int cols, PixelYUV *input, int row, int col) {
	if (row >= 0 && row < rows && col >= 0 && col < cols) {
		return input[row * cols + col];
	} else if (row < 0 && col < 0) {
		// top left corner
		return input[0 * cols + 0];
	} else if (row < 0 && col >= cols) {
		// top right corner
		return input[0 * cols + (cols - 1)];
	} else if (row >= rows && col < 0) {
		// bottom left corner
		return input[(rows - 1) * cols + 0];
	} else if (row >= rows && col >= cols) {
		// bottom right corner
		return input[(rows - 1) * cols + (cols - 1)];
	} else if (row < 0) {
		// top edge
		return input[0 * cols + col];
	} else if (row >= rows) {
		// bottom edge
		return input[(rows - 1) * cols + col];
	} else if (col < 0) {
		// left edge
		return input[row * cols + 0];
	} else if (col >= cols) {
		// right edge
		return input[row * cols + (cols - 1)];
	} else {
		// this should never happen
		assert(false);
		return {0, 0, 0};
	}
}

PixelRGB get_RGB(unsigned int rows, unsigned int cols, PixelRGB *input, int row, int col) {
	if (row >= 0 && row < rows && col >= 0 && col < cols) {
		return input[row * cols + col];
	} else if (row < 0 && col < 0) {
		// top left corner
		return input[0 * cols + 0];
	} else if (row < 0 && col >= cols) {
		// top right corner
		return input[0 * cols + (cols - 1)];
	} else if (row >= rows && col < 0) {
		// bottom left corner
		return input[(rows - 1) * cols + 0];
	} else if (row >= rows && col >= cols) {
		// bottom right corner
		return input[(rows - 1) * cols + (cols - 1)];
	} else if (row < 0) {
		// top edge
		return input[0 * cols + col];
	} else if (row >= rows) {
		// bottom edge
		return input[(rows - 1) * cols + col];
	} else if (col < 0) {
		// left edge
		return input[row * cols + 0];
	} else if (col >= cols) {
		// right edge
		return input[row * cols + (cols - 1)];
	} else {
		// this should never happen
		assert(false);
		return {0, 0, 0};
	}
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
void expand_pixel(unsigned int rows, unsigned int cols, PixelRGB *inputRGB, PixelYUV *inputYUV, PixelRGB *output, unsigned int scaleFactor, unsigned int row, unsigned int col) {
	int edgeBR;
	int edgeBL;
	int edgeTL;
	int edgeTR;

	// edge detection
	PixelYUV A = get_YUV(rows, cols, inputYUV, row-1, col-1);
	PixelYUV B = get_YUV(rows, cols, inputYUV, row-1, col);
	PixelYUV C = get_YUV(rows, cols, inputYUV, row-1, col+1);
	PixelYUV D = get_YUV(rows, cols, inputYUV, row, col-1);
	PixelYUV E = get_YUV(rows, cols, inputYUV, row, col);
	PixelYUV F = get_YUV(rows, cols, inputYUV, row, col+1);
	PixelYUV G = get_YUV(rows, cols, inputYUV, row+1, col-1);
	PixelYUV H = get_YUV(rows, cols, inputYUV, row+1, col);
	PixelYUV I = get_YUV(rows, cols, inputYUV, row+1, col+1);

	PixelYUV A1 = get_YUV(rows, cols, inputYUV, row-2, col-1);
	PixelYUV B1 = get_YUV(rows, cols, inputYUV, row-2, col);
	PixelYUV C1 = get_YUV(rows, cols, inputYUV, row-2, col+1);
	PixelYUV A0 = get_YUV(rows, cols, inputYUV, row-1, col-2);
	PixelYUV C4 = get_YUV(rows, cols, inputYUV, row-1, col+2);
	PixelYUV D0 = get_YUV(rows, cols, inputYUV, row, col-2);
	PixelYUV F4 = get_YUV(rows, cols, inputYUV, row, col+2);
	PixelYUV G0 = get_YUV(rows, cols, inputYUV, row+1, col-2);
	PixelYUV I4 = get_YUV(rows, cols, inputYUV, row+1, col+2);
	PixelYUV G5 = get_YUV(rows, cols, inputYUV, row+2, col-1);
	PixelYUV H5 = get_YUV(rows, cols, inputYUV, row+2, col);
	PixelYUV I5 = get_YUV(rows, cols, inputYUV, row+2, col+1);

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
	PixelRGB *out = &output[row * out_cols * scaleFactor + col * scaleFactor];

	PixelRGB E_rgb = get_RGB(rows, cols, inputRGB, row, col);
	for (int r = 0; r < scaleFactor; r++) {
		for (int c = 0; c < scaleFactor; c++) {
			out[r * out_cols + c] = E_rgb;
		}
	}

	if (edgeBR) {
		PixelRGB F_rgb = get_RGB(rows, cols, inputRGB, row, col+1);
		PixelRGB H_rgb = get_RGB(rows, cols, inputRGB, row+1, col);
		PixelRGB newColor = d(E, F) <= d(E, H) ? F_rgb : H_rgb;

		if (YUV_equals(F, G) && YUV_equals(C, H)) {
			BR_INT2_B(scaleFactor, out, out_cols, newColor);
			BR_INT2_R(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(F, G))
			BR_INT2_B(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(C, H))
			BR_INT2_R(scaleFactor, out, out_cols, newColor);
		else
			BR_INT1(scaleFactor, out, out_cols, newColor);
	}

	if (edgeBL) {
		PixelRGB D_rgb = get_RGB(rows, cols, inputRGB, row, col-1);
		PixelRGB H_rgb = get_RGB(rows, cols, inputRGB, row+1, col);
		PixelRGB newColor = d(D, E) <= d(E, H) ? D_rgb : H_rgb;

		if (YUV_equals(D, I) && YUV_equals(A, H)) {
			BL_INT2_B(scaleFactor, out, out_cols, newColor);
			BL_INT2_L(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(D, I))
			BL_INT2_B(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(A, H))
			BL_INT2_L(scaleFactor, out, out_cols, newColor);
		else
			BL_INT1(scaleFactor, out, out_cols, newColor);
	}

	if (edgeTL) {
		PixelRGB B_rgb = get_RGB(rows, cols, inputRGB, row-1, col);
		PixelRGB D_rgb = get_RGB(rows, cols, inputRGB, row, col-1);
		PixelRGB newColor = d(B, E) <= d(D, E) ? B_rgb : D_rgb;

		if (YUV_equals(C, D) && YUV_equals(B, G)) {
			TL_INT2_T(scaleFactor, out, out_cols, newColor);
			TL_INT2_L(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(C, D))
			TL_INT2_T(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(B, G))
			TL_INT2_L(scaleFactor, out, out_cols, newColor);
		else
			TL_INT1(scaleFactor, out, out_cols, newColor);
	}

	if (edgeTR) {
		PixelRGB B_rgb = get_RGB(rows, cols, inputRGB, row-1, col);
		PixelRGB F_rgb = get_RGB(rows, cols, inputRGB, row, col+1);
		PixelRGB newColor = d(B, E) <= d(E, F) ? B_rgb : F_rgb;

		if (YUV_equals(A, F) && YUV_equals(B, I)) {
			TR_INT2_T(scaleFactor, out, out_cols, newColor);
			TR_INT2_R(scaleFactor, out, out_cols, newColor);
		} else if (YUV_equals(A, F))
			TR_INT2_T(scaleFactor, out, out_cols, newColor);
		else if (YUV_equals(B, I))
			TR_INT2_R(scaleFactor, out, out_cols, newColor);
		else
			TR_INT1(scaleFactor, out, out_cols, newColor);
	}

}

#endif

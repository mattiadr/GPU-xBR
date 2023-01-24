#ifndef XBR_INTERPOLATIONS
#define XBR_INTERPOLATIONS

#include "img_utils.h"

#define INT1_even 0.5
#define INT1_odd_s 0.125
#define INT1_odd_l 0.875

#define INT2_s 0.25
#define INT2_l 0.75

/**
 * Common
 **/

__device__ PixelRGB mix_colors(PixelRGB a, PixelRGB b, double percent) {
	PixelRGB ret;
	ret.R = a.R * (1 - percent) + b.R * percent;
	ret.G = a.G * (1 - percent) + b.G * percent;
	ret.B = a.B * (1 - percent) + b.B * percent;
	return ret;
}

/**
 * Bottom Right
 **/

__device__ void BR_INT1(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT1_even);
		break;
	case 3:
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT1_odd_s);
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT1_odd_s);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT1_odd_l);
		break;
	case 4:
		out[2 * out_cols + 3] = mix_colors(out[2 * out_cols + 3], newColor, INT1_even);
		out[3 * out_cols + 2] = mix_colors(out[3 * out_cols + 2], newColor, INT1_even);
		out[3 * out_cols + 3] = newColor;
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void BR_INT2_B(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_l);
		break;
	case 3:
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_s);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_s);
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_l);
		out[2 * out_cols + 2] = newColor;
		break;
	case 4:
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_s);
		out[2 * out_cols + 3] = mix_colors(out[2 * out_cols + 3], newColor, INT2_l);
		out[3 * out_cols + 0] = mix_colors(out[3 * out_cols + 0], newColor, INT2_s);
		out[3 * out_cols + 1] = mix_colors(out[3 * out_cols + 1], newColor, INT2_l);
		out[3 * out_cols + 2] = newColor;
		out[3 * out_cols + 3] = newColor;
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void BR_INT2_R(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_s);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_l);
		break;
	case 3:
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_s);
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_l);
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_s);
		out[2 * out_cols + 2] = newColor;
		break;
	case 4:
		out[0 * out_cols + 3] = mix_colors(out[0 * out_cols + 3], newColor, INT2_s);
		out[1 * out_cols + 3] = mix_colors(out[1 * out_cols + 3], newColor, INT2_l);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_s);
		out[2 * out_cols + 3] = newColor;
		out[3 * out_cols + 2] = mix_colors(out[3 * out_cols + 2], newColor, INT2_l);
		out[3 * out_cols + 3] = newColor;
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

/**
 * Bottom Left
 **/

__device__ void BL_INT1(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT1_even);
		break;
	case 3:
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT1_odd_s);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT1_odd_l);
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT1_odd_s);
		break;
	case 4:
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT1_even);
		out[3 * out_cols + 0] = newColor;
		out[3 * out_cols + 1] = mix_colors(out[3 * out_cols + 1], newColor, INT1_even);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void BL_INT2_B(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_s);
		break;
	case 3:
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_s);
		out[2 * out_cols + 0] = newColor;
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_l);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_s);
		break;
	case 4:
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_s);
		out[3 * out_cols + 0] = newColor;
		out[3 * out_cols + 1] = newColor;
		out[3 * out_cols + 2] = mix_colors(out[3 * out_cols + 2], newColor, INT2_l);
		out[3 * out_cols + 3] = mix_colors(out[3 * out_cols + 3], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void BL_INT2_L(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		break;
	case 3:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 0] = newColor;
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 0] = newColor;
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_s);
		out[3 * out_cols + 0] = newColor;
		out[3 * out_cols + 1] = mix_colors(out[3 * out_cols + 1], newColor, INT2_l);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

/**
 * Top Left
 **/

__device__ void TL_INT1(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT1_even);
		break;
	case 3:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT1_odd_l);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT1_odd_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT1_odd_s);
		break;
	case 4:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT1_even);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT1_even);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void TL_INT2_T(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_l);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_s);
		break;
	case 3:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = newColor;
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_l);
		out[0 * out_cols + 3] = mix_colors(out[0 * out_cols + 3], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void TL_INT2_L(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_l);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_s);
		break;
	case 3:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[1 * out_cols + 0] = newColor;
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_s);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_l);
		out[3 * out_cols + 0] = mix_colors(out[3 * out_cols + 0], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

/**
 * Top Right
 **/

__device__ void TR_INT1(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT1_even);
		break;
	case 3:
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT1_odd_s);
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT1_odd_s);
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT1_odd_l);
		break;
	case 4:
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT1_even);
		out[1 * out_cols + 3] = mix_colors(out[1 * out_cols + 3], newColor, INT1_even);
		out[0 * out_cols + 3] = newColor;
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void TR_INT2_T(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		break;
	case 3:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[0 * out_cols + 2] = newColor;
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[0 * out_cols + 2] = newColor;
		out[0 * out_cols + 3] = newColor;
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_s);
		out[1 * out_cols + 3] = mix_colors(out[1 * out_cols + 3], newColor, INT2_l);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void TR_INT2_R(unsigned int scaleFactor, PixelRGB *out, unsigned int out_cols, PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_s);
		break;
	case 3:
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_s);
		out[0 * out_cols + 2] = newColor;
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_l);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_l);
		out[0 * out_cols + 3] = newColor;
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_s);
		out[1 * out_cols + 3] = newColor;
		out[2 * out_cols + 3] = mix_colors(out[2 * out_cols + 3], newColor, INT2_l);
		out[3 * out_cols + 3] = mix_colors(out[3 * out_cols + 3], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

#endif

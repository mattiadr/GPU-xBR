#ifndef XBR_INTERPOLATIONS
#define XBR_INTERPOLATIONS

#include "img_utils.h"

#define INT1_even 0.5
#define INT1_odd_s 0.125
#define INT1_odd_l 0.875

#define INT2_s 0.25
#define INT2_l 0.75
#define INT2_ss 0.5
#define INT2_ll 0.8

#define USE_HALF 0

/**
 * Common
 **/

#if USE_HALF
#include "cuda_fp16.h"

__device__ PixelRGB mix_colors(const PixelRGB a, const PixelRGB b, const half percent) {
	PixelRGB ret;
	ret.R = __half2ushort_rn(__hfma(a.R, __hsub(1, percent), __hmul(b.R, percent)));
	ret.G = __half2ushort_rn(__hfma(a.G, __hsub(1, percent), __hmul(b.G, percent)));
	ret.B = __half2ushort_rn(__hfma(a.B, __hsub(1, percent), __hmul(b.B, percent)));
	return ret;
}
#else
__device__ PixelRGB mix_colors(const PixelRGB a, const PixelRGB b, const float percent) {
	PixelRGB ret;
	ret.R = a.R * (1 - percent) + b.R * percent;
	ret.G = a.G * (1 - percent) + b.G * percent;
	ret.B = a.B * (1 - percent) + b.B * percent;
	return ret;
}
#endif

/**
 * Bottom Right
 **/

__device__ void BR_INT1(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void BR_INT2_BR(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_ll);
		break;
	case 3:
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_s);
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_l);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_s);
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_l);
		out[2 * out_cols + 2] = newColor;
		break;
	case 4:
		out[0 * out_cols + 3] = mix_colors(out[0 * out_cols + 3], newColor, INT2_s);
		out[1 * out_cols + 3] = mix_colors(out[1 * out_cols + 3], newColor, INT2_l);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_ss);
		out[2 * out_cols + 3] = newColor;
		out[3 * out_cols + 0] = mix_colors(out[3 * out_cols + 0], newColor, INT2_s);
		out[3 * out_cols + 1] = mix_colors(out[3 * out_cols + 1], newColor, INT2_l);
		out[3 * out_cols + 2] = newColor;
		out[3 * out_cols + 3] = newColor;
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void BR_INT2_B(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void BR_INT2_R(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void BL_INT1(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void BL_INT2_BL(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_ll);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_s);
		break;
	case 3:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 0] = newColor;
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_l);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 0] = newColor;
		out[2 * out_cols + 1] = mix_colors(out[2 * out_cols + 1], newColor, INT2_ss);
		out[3 * out_cols + 0] = newColor;
		out[3 * out_cols + 1] = newColor;
		out[3 * out_cols + 2] = mix_colors(out[3 * out_cols + 2], newColor, INT2_l);
		out[3 * out_cols + 3] = mix_colors(out[3 * out_cols + 3], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void BL_INT2_B(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void BL_INT2_L(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void TL_INT1(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void TL_INT2_TL(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_ll);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_s);
		break;
	case 3:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_s);
		out[1 * out_cols + 0] = mix_colors(out[1 * out_cols + 0], newColor, INT2_l);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = newColor;
		out[0 * out_cols + 1] = newColor;
		out[0 * out_cols + 2] = mix_colors(out[0 * out_cols + 2], newColor, INT2_l);
		out[0 * out_cols + 3] = mix_colors(out[0 * out_cols + 3], newColor, INT2_s);
		out[1 * out_cols + 0] = newColor;
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_ss);
		out[2 * out_cols + 0] = mix_colors(out[2 * out_cols + 0], newColor, INT2_l);
		out[3 * out_cols + 0] = mix_colors(out[3 * out_cols + 0], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void TL_INT2_T(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void TL_INT2_L(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void TR_INT1(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void TR_INT2_TR(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
	switch (scaleFactor) {
	case 2:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_ll);
		out[1 * out_cols + 1] = mix_colors(out[1 * out_cols + 1], newColor, INT2_s);
		break;
	case 3:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[0 * out_cols + 2] = newColor;
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_l);
		out[2 * out_cols + 2] = mix_colors(out[2 * out_cols + 2], newColor, INT2_s);
		break;
	case 4:
		out[0 * out_cols + 0] = mix_colors(out[0 * out_cols + 0], newColor, INT2_s);
		out[0 * out_cols + 1] = mix_colors(out[0 * out_cols + 1], newColor, INT2_l);
		out[0 * out_cols + 2] = newColor;
		out[0 * out_cols + 3] = newColor;
		out[1 * out_cols + 2] = mix_colors(out[1 * out_cols + 2], newColor, INT2_ss);
		out[1 * out_cols + 3] = newColor;
		out[2 * out_cols + 3] = mix_colors(out[2 * out_cols + 3], newColor, INT2_l);
		out[3 * out_cols + 3] = mix_colors(out[3 * out_cols + 3], newColor, INT2_s);
		break;
	// default:
		// fprintf(stderr, "Not implemented error: scaleFactor %d not supported.\n", scaleFactor);
	}
}

__device__ void TR_INT2_T(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

__device__ void TR_INT2_R(const unsigned int scaleFactor, PixelRGB *out, const unsigned int out_cols, const PixelRGB newColor) {
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

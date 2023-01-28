#ifndef IMG_UTILS
#define IMG_UTILS

typedef struct __align__(4) {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} PixelRGB;

typedef struct __align__(4) {
	unsigned char Y;
	unsigned char U;
	unsigned char V;
} PixelYUV;

__device__ PixelYUV rgb_to_yuv(PixelRGB input) {
	PixelYUV output;
	output.Y =   0.299f * input.R + 0.587f * input.G + 0.114f * input.B;
	output.U = - 0.169f * input.R - 0.331f * input.G + 0.500f * input.B;
	output.V =   0.500f * input.R - 0.419f * input.G - 0.081f * input.B;
	return output;
}

#endif

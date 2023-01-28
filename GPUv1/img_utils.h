#ifndef IMG_UTILS
#define IMG_UTILS

typedef struct {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} PixelRGB;

typedef struct __align__(4) {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} PixelRGB4;

typedef struct __align__(4) {
	unsigned char Y;
	unsigned char U;
	unsigned char V;
} PixelYUV;

__device__ PixelYUV rgb_to_yuv(PixelRGB input) {
	PixelYUV output;
	output.Y =   0.299 * input.R + 0.587 * input.G + 0.114 * input.B;
	output.U = - 0.169 * input.R - 0.331 * input.G + 0.500 * input.B;
	output.V =   0.500 * input.R - 0.419 * input.G - 0.081 * input.B;
	return output;
}

__device__ PixelRGB4 rgb_to_rgb4(PixelRGB p) {
	return (PixelRGB4) {.R = p.R, .G = p.G, .B = p.B};
}

__device__ PixelRGB rgb4_to_rgb(PixelRGB4 p) {
	return (PixelRGB) {.R = p.R, .G = p.G, .B = p.B};
}

#endif

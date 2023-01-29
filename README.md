# GPU-xBR

PoliTO GPU Programming project (xBR pixel art upscaling algorithm)

- D'Andrea Giuseppe s303378
- De Rosa Mattia s303379

## How to compile

Dependencies:
- opencv4

```bash
nvcc `pkg-config --libs --cflags opencv4` GPUv0/main.cu -o main.out -maxrregcount 32 -O3
```

## How to run

`./main.out scale_factor input_file output_file`

## How to re-encode videos to H.264

`ffmpeg -i input.avi -c:v libx264 -qp 0 output.avi`

## Versions

- CPU: cpu version with no optimizations
- GPUv0: naive GPU implementation with no optimizations
- GPUv1:

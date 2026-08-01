# array-morphisms-blas

System-BLAS-backed `blas-backend` adapter for
[array-morphisms](https://github.com/iraikov/array-morphisms).

This egg exists separately from the core `array-morphisms` in order to avoid introducing a dependency on blas.
`array-morphisms` provides a default backend based on the `microBLAS` header-only library.
Install `array-morphisms-blas` only when you want system-BLAS acceleration (OpenBLAS, MKL, etc., via the CHICKEN `blas` egg).

## Installation

```bash
chicken-install array-morphisms-blas
```

Requires the [`blas`](https://wiki.call-cc.org/eggref/5/blas) egg and a system BLAS library
(e.g. OpenBLAS) to already be resolvable/installed.

## Which BLAS implementation actually gets used (Debian/Ubuntu)

The CHICKEN `blas` egg's build script does not choose a specific BLAS implementation itself; its
`build.scm` just probes a fixed list of generic candidates (`<cblas.h>` + `-lblas -lm` first,
falling back to `-lcblas`, GSL's cblas, ATLAS, etc.). What `-lblas` actually resolves to is decided
entirely by the OS's `update-alternatives` system, not by anything in the egg.

On Debian-based systems, installing OpenBLAS's dev packages (`libopenblas-pthread-dev`,
`libopenblas-openmp-dev`, and/or `libopenblas-serial-dev`) registers additional alternatives for
both `libblas.so-x86_64-linux-gnu` (the dev-time header + `.so` used at link time)
and `libblas.so.3-x86_64-linux-gnu` (the runtime SONAME actually
loaded at run time), each at a higher priority than the reference Netlib BLAS implementation:

| Provider | Priority | Threading |
|---|---|---|
| `openblas-pthread` | 100 (highest) | raw pthreads |
| `openblas-openmp` | 95 | OpenMP (`libgomp`) |
| `openblas-serial` | 90 | none |
| reference (Netlib) `blas` | 10 | none |

Both alternative groups default to "auto" mode, so the highest-priority option (`openblas-pthread`)
is selected automatically the moment the corresponding packages are installed.
If OpenBLAS is installed on the machine, both the plain `blas` egg and this egg
already link against and run on OpenBLAS.

To check or change which implementation is active:

```bash
update-alternatives --display libblas.so.3-x86_64-linux-gnu   # runtime selection
update-alternatives --display libblas.so-x86_64-linux-gnu     # dev/link-time selection
update-alternatives --config libblas.so.3-x86_64-linux-gnu    # switch interactively
```

## Usage

```scheme
(import array-morphisms-blas-exec)
(import array-morphisms-blas-egg-backend)

(register-blas-backend! (make-blas-egg-backend))
```

Registering a backend overrides whatever backend `array-morphisms` auto-registered by default
(the built-in `microBLAS` tier); the last `register-blas-backend!` call wins.

## Running OpenBLAS in parallel

OpenBLAS parallelizes automatically by default, using all CPU cores it detects, i.e. nothing needs
to be set for multi-threaded execution to happen at all. The environment variables below are for
*tuning*, not for switching parallelism on:

- **`OPENBLAS_NUM_THREADS=<n>`**: bounds the thread count; works regardless of which OpenBLAS
  variant is linked (pthread, openmp, or serial). This is the one to set if unsure which variant
  is active.
- **`OMP_NUM_THREADS=<n>`**: affects the `openblas-openmp` variant specifically (and any other
  OpenMP-using code sharing the process); has no effect on the pthread variant.

```bash
OPENBLAS_NUM_THREADS=4 csi -s your-script.scm
```

## What it provides

`make-blas-egg-backend` constructs a `blas-backend` record (from `array-morphisms-blas-exec`)
that routes gemm/gemv/dot/axpy through the `blas` egg's `dgemm!`/`sgemm!`/etc., and additionally
provides fused im2col+gemm+bias "hot kernel" hooks (both NCHW and NHWC layouts) for
`array-morphisms`'s conv2d forward/backward paths, backed by the C kernels in `kernels/im2col.c`.

## License

LGPL-3, the same as `array-morphisms`.

# array-morphisms-blas

System-BLAS-backed `blas-backend` adapter for
[array-morphisms](https://github.com/iraikov/array-morphisms).

This egg exists separately from the core `array-morphisms` egg because CHICKEN's egg system
has no way to make a component's dependency on the `blas` egg conditional: listing `blas` in
`array-morphisms.egg` would force every `array-morphisms` user to install the `blas` egg (and a
system BLAS library) even if they only want the dependency-free `microBLAS` default backend that
ships with the core egg. Install `array-morphisms-blas` only when you want system-BLAS
acceleration (OpenBLAS, MKL, etc., via the CHICKEN `blas` egg).

## Installation

```bash
chicken-install array-morphisms-blas
```

Requires the [`blas`](https://wiki.call-cc.org/eggref/5/blas) egg and a system BLAS library
(e.g. OpenBLAS) to already be resolvable/installed.

## Which BLAS implementation actually gets used (Debian/Ubuntu)

> **Provisional note**, written after investigating why this egg's performance changed once
> OpenBLAS packages were installed on a Debian-derived dev machine. Worth re-checking on other
> distros/setups.

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

Verified directly:
- `ldd` on the built `.so` shows `libblas.so.3 → libopenblas.so.0`.
- `nm -D libblas.so.3 | grep cblas_ddot` finds the symbol (the reference implementation alone
  would not export a CBLAS interface).
- Benchmarked: a 2048×2048 `sgemm` goes from ~105 GFLOP/s (1 thread) to ~310 GFLOP/s (4 threads).

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

Worth setting explicitly when running multiple BLAS-heavy processes concurrently on the same
machine (so they don't collectively oversubscribe the CPU), or when you want a fixed, reproducible
thread count for benchmarking rather than "whatever this machine's core count happens to be."

## What it provides

`make-blas-egg-backend` constructs a `blas-backend` record (from `array-morphisms-blas-exec`)
that routes gemm/gemv/dot/axpy through the `blas` egg's `dgemm!`/`sgemm!`/etc., and additionally
provides fused im2col+gemm+bias "hot kernel" hooks (both NCHW and NHWC layouts) for
`array-morphisms`'s conv2d forward/backward paths, backed by the C kernels in `kernels/im2col.c`.

## License

LGPL-3, the same as `array-morphisms`.

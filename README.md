# CUDA Concurrent Kernel Execution Framework (cckef)

A framework for executing multiple CUDA kernels concurrently. It provides a set of examples and benchmarks to explore and test concurrent execution on NVIDIA GPUs. This framework is designed to help understand and optimize the usage of CUDA streams and the Multi-Process Service (MPS) for overlapping computation and data transfers.

## Prerequisites

Before you begin, ensure you have the following installed:
*   An NVIDIA GPU with CUDA support
*   [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
*   A C++ compiler (e.g., GCC)
*   `make`

## Building the Project

To build all the test applications and benchmarks, run the `make` command in the root directory of the project:

```bash
make
```

This will compile the source files and create executables in the root directory.

To build a specific test, you can pass it as an argument to `make`. For example:

```bash
make concTest
```

To clean up the build files and executables, run:

```bash
make clean
```

## Running Tests and Benchmarks

The framework includes several tests and benchmarks to demonstrate and measure concurrent kernel execution.

### Main Executables

*   `soloTest`: Executes a sequence of kernels from different applications one by one.
*   `concTest`: Executes multiple kernels concurrently using CUDA streams.
*   `ckeTest`: A test for Concurrent Kernel Execution.
*   `mpsBench`: A benchmark for NVIDIA's Multi-Process Service (MPS).
*   `soloBench`: A benchmark for single kernel execution performance.
*   `example`: A simple example to demonstrate basic functionality.

You can run them from the command line. For example:

```bash
./concTest
```

## Project Structure

The project is organized into several directories:

*   `BlackScholes/`: An implementation of the Black-Scholes model for option pricing.
*   `dummy/`: A dummy task for testing purposes.
*   `matrixMul/`: A matrix multiplication example.
*   `memBench/`: A memory benchmark.
*   `PathFinder/`: An implementation of a pathfinding algorithm.
*   `vectorAdd/`: A simple vector addition example.
*   `tasks/`: Contains the core CUDA task management classes.
*   `profile/`: Profiling utilities.

Each application directory (e.g., `BlackScholes/`, `matrixMul/`) contains the specific kernel implementation (`*Kernel.cuh`) and the task definition (`*Task.cu`).

## Our papers:
If you find this code useful in your research, please consider citing:

[FlexSched: Efficient scheduling techniques for concurrent kernel execution on GPUs](https://doi.org/10.1007/s11227-021-03819-z)

```
@ARTICLE{López-Albelda202243,
	author = {López-Albelda, Bernabé and Castro, Francisco M. and González-Linares, José M. and Guil, Nicolás},
	title = {FlexSched: Efficient scheduling techniques for concurrent kernel execution on GPUs},
	year = {2022},
	journal = {Journal of Supercomputing},
	volume = {78},
	number = {1},
	pages = {43 – 71},
	doi = {10.1007/s11227-021-03819-z},
}
```
# Single-particle tracking (SPT)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This Python library is designed to enhance the performance of single-particle tracking (SPT) algorithms by leveraging Cython. The algorithm is based on the method described in the paper:

**Jaqaman, K., Loerke, D., Mettlen, M. et al. Robust single-particle tracking in live-cell time-lapse sequences. Nat Methods 5, 695–702 (2008).** [https://doi.org/10.1038/nmeth.1237](https://doi.org/10.1038/nmeth.1237)

## Features

- **Cython-based Compilation:** Significant performance improvements over the pure Python implementation of the tracking algorithm by compiling key parts using Cython.
- **Ease of Integration:** The library is designed to integrate seamlessly with existing Python codebases.
- **Customizable:** Users can easily customize and extend the functionality to suit their specific needs.

## Installation

To install this library, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/franciscopalmeromoya/single-particle-tracking.git
cd single-particle-tracking
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Alternatively, if you have a `setup.py` configured with `cythonize`, you can install it directly using:

```bash
pip install .
```

## Usage

Here is a basic example of how to use the library in your project:

```python
import spt

# Initialize the tracking algorithm with your data
tracker = spt.Tracker(skip_frames=3, max_dist=2)

# Run the enhanced tracking algorithm
results = tracker.track(data)
```

### Example Workflow

1. **Prepare Your Data:** Load your time-lapse sequence data into a format compatible with the algorithm. The data must be a dataframe with columns ['frame', 'x', 'y'];
2. **Initialize the Tracker:** Use the `spt.Tracker` class provided by the library.
3. **Run the Tracking:** Execute the `track` method to track particles.

## Performance

This library leverages Cython to compile performance-critical sections of the SPT algorithm, resulting in faster execution times. Benchmarks have shown significant speedups compared to the original Python implementation, especially for large datasets.

## Contributing

We welcome contributions from the community! If you encounter bugs, have feature requests, or want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Commit your changes and push the branch.
4. Create a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## References

If you use this library in your research, please cite the original paper:

Jaqaman, K., Loerke, D., Mettlen, M. et al. Robust single-particle tracking in live-cell time-lapse sequences. Nat Methods 5, 695–702 (2008). [https://doi.org/10.1038/nmeth.1237](https://doi.org/10.1038/nmeth.1237)

---

## Contact

For questions, feedback, or support, please open an issue in the repository or contact me by email.


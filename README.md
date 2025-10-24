# [NeurIPS 2025] Stab-SGD: Noise-Adaptivity in Smooth Optimization with Stability Ratios

by **David A. R. Robin**, **Killian Bakong**, and **Kevin Scaman**.

The root folder contains the framework to run comparisons of all algorithms
on the convex problems presented in Section 3.
The `resnet-experiment/` folder contains the source code to run the experiments
of Section 4. See the [associated readme](resnet-experiment/README.md) for details.

## Setup

Use `make install` or `pip -r requirements.txt` to install and upgrade all required python packages.


## Run experiments

Use `make run`, and possibly modify HORIZON, SEEDCOUNT or ALGORITHMS variables in `Makefile`.

Compile all results across seeds into plottable csv files with `make plot`.
To show plots on top of the csv compilation, remove the `--noshow` argument in `make plot` target in `Makefile`.

The individual results (per seed) can be found in `measurements/` directory, and compiled
files with means / medians at each time or performance as a function of tuning horizon are
found in the `export/` directory.

## Reference
- D. A. R. Robin, K. Bakong, K. Scaman. [*Stab-SGD: Noise-Adaptivity in Smooth Optimization with Stability Ratios*](https://openreview.net/forum?id=JRFMzQnYXl). NeurIPS 2025.

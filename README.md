# Contextual DMControl Benchmark
Benchmark for generalization in continuous control from pixels, based on [DMControl Generalization Benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark).
It is adapted to allow for direct control of the _contexts_ by supplying at the start of an experiment a training set and testing set of colours, video backgrounds and initial physics engine states. 

## Contexts

The DMControl Generalization Benchmark provides full control for creating benchmarks for visual generalization to random colors, video backgrounds and initial states.

Example colors and video backgrounds can be found in ```cdmc/env/data/``` and ```cdmc/generate_contexts.py``` shows an example of how to create training and testing sets.

Using an empty context set (in ```empty.json```) will run on the default colors and background and sample initial physics states from the full distribution (default DMC behaviour). 


## Algorithms

This repository contains implementations of the following algorithms in a unified framework:

- [SVEA (Hansen et al., 2021)](https://arxiv.org/abs/2107.00644)
- [SODA (Hansen and Wang, 2021)](https://arxiv.org/abs/2011.13389)
- [PAD (Hansen et al., 2020)](https://arxiv.org/abs/2007.04309)
- [DrQ (Kostrikov et al., 2020)](https://arxiv.org/abs/2004.13649)
- [RAD (Laskin et al., 2020)](https://arxiv.org/abs/2004.14990)
- [CURL (Srinivas et al., 2020)](https://arxiv.org/abs/2004.04136)
- [SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)

using standardized architectures and hyper-parameters, wherever applicable.

## Setup
First install mujoco dependency:
1. Download old version of [mujoco](https://www.roboti.us/download.html) and paste it in your home .mujoco folder
2. Download free mujoco [license](https://www.roboti.us/license.html) and paste it in your home .mujoco folder

Then, we assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:

```
conda env create -f setup/conda.yaml
conda activate dmcgb
sh setup/install_envs.sh
```

## Datasets
Part of this repository relies on external datasets. SODA uses the [Places](http://places2.csail.mit.edu/download.html) dataset for data augmentation, which can be downloaded by running

```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

The `video_easy` data was proposed in [PAD](https://github.com/nicklashansen/policy-adaptation-during-deployment), and the `video_hard` data uses a subset of the [RealEstate10K](https://google.github.io/realestate10k/) dataset for background rendering. All test environments (including video files) are included in this repository, namely in the `cdmc/env/` directory.


## Training & Evaluation

The `scripts` directory contains training and evaluation bash scripts for all the included algorithms. Alternatively, you can call the python scripts directly, e.g. for training call

```
python3 cdmc/train.py \
  --algorithm sac \
  --seed 0 \
  --train_context_file empty.json \
	--test_context_file empty.json
```

to run SAC on the default task, `walker_walk`.

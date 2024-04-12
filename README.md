# MidLLaMAI

Steps & Dependencies:

Conda environment Python3.11

conda install nvidia/label/cuda-12.1.0::cuda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Then install whatever pip dependencies you get as complaints when running the below scripts.

For chatting:
python prompt.py

For benchmarking:
python benchmark.py
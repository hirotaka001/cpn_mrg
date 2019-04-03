#!/bin/bash 

~/anaconda3/envs/trt5_cpn/bin/python setup.py build_ext --inplace
rm -rf build/
mv ./nms/cpu_nms*.so .
mv ./nms/gpu_nms*.so .

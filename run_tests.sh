#!/bin/sh
# Builds tests.cpp against OpenCV and runs it from the project folder
# Set OPENCV_PREFIX if OpenCV is somewhere other than the cv conda environment
cd "$(dirname "$0")" || exit 1
PREFIX="${OPENCV_PREFIX:-$HOME/miniconda3/envs/cv}"
OUT="${TMPDIR:-/tmp}/pool_tests"
clang++ -std=c++17 -I"$PREFIX/include/opencv4" tests.cpp -o "$OUT" \
    -L"$PREFIX/lib" -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
    -Wl,-rpath,"$PREFIX/lib" && "$OUT"

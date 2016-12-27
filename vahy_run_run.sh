#!/usr/bin/env bash

echo "Compiling the project"
mkdir build
cd build
cmake ..
make -j8
cd ..
echo "Compilation done"

echo "Downloading sample dataset"
mkdir samples
cd samples
../data/download.sh 2015-01-01 2015-01-04
cd ..
echo "Downloading done"

mkdir result

echo "Compute PCA"
build/pca-matrix samples result/samples.pca
echo "Done computing PCA"

echo "Learn the network"
build/lstm --data="samples" --pca="result/samples.pca" --learn=1 --s="result/network" --ite=1
echo "Learning done"

echo "Evaluate the network"
build/lstm --data="samples" --pca="result/samples.pca" --learn=0 --l="result/network" --o="result/output.jpg"
echo "Evaluating done - see directory result"

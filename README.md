# pv021_project

Building is done by cmake. To successfully build the project you need to have c++14 capable compiler and opencv-2.4 library installed.

Following steps could be useful for usage of neural network:

## Download data set
In order to download data set use 'download.sh' script from directory where you want the data to be stored. 'download.sh' takes two arguments of time interval of samples. Example:

'''
    ./download.sh 2015-1-1 2015-12-31
'''

would download the whole year 2015.

If you want to change the sample rate, rewrite the line 22 in download script to desired rate in minutes.

## PCA computation
'pca-matrix' binary is used to compute PCA matrix from dataset. Usage example:

'''
    ./pca-matrix my-dataset my-dataset.pca
'''
Where my-dataset is directory located in same directory as 'pca-matrix' binary and contains
only downloaded data.

## Learning
To run lstm network in learning mode run following command:
'''
    ./lstm --data=[data set path] --pca=[path to pca matrix] --ite=[number of iterations]
           --s=[save network path]
'''

Before run be sure, that the dimension of pca matrix is same as 'input_size' in 'main.cpp'.

If you dont know the pca matrix size, run the 'lstm' binary with given matrix and change the
'input_size' on line 22 in 'main.cpp'.

## Evaluating
To run lstm network in evaluation mode run following command:
'''
    ./lstm --data=[data set path] --pca=[path to pca matrix] --l=[network path] --o=[result.jpg]
'''

The network will load first 7 images from data set and comput the following one.

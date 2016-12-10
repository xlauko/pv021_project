#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "serialization.hpp"
#include "filter.hpp"

using Image = cv::Mat;

Image reshape_images( std::vector< Image > & data );
cv::PCA compute_pca( std::vector< cv::String > filenames );

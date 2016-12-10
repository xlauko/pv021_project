#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "serialization.hpp"
#include "filter.hpp"

using Image = cv::Mat;

Image to_pca( Image & img, cv::PCA & pca );
Image to_pca( Image & img, std::string & pca_path );
Image from_pca( const std::string & path, cv::PCA & pca );
Image from_pca( const std::string & path, const std::string & pca_path );

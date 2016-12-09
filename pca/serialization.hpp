#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <tuple>

namespace serial {

using Image = cv::Mat;

static void save_pca(const std::string & file_name, cv::PCA pca ) {
    cv::FileStorage fs( file_name, cv::FileStorage::WRITE );
    fs << "mean" << pca.mean;
    fs << "e_vectors" << pca.eigenvectors;
    fs << "e_values" << pca.eigenvalues;
    fs.release();
}

static cv::PCA load_pca( const std::string & file_name ) {
    cv::PCA pca;
    cv::FileStorage fs( file_name, cv::FileStorage::READ );
    fs["mean"] >> pca.mean ;
    fs["e_vectors"] >> pca.eigenvectors ;
    fs["e_values"] >> pca.eigenvalues ;
    fs.release();
    return pca;
}

static void save_img_pca( const std::string & out_path, Image & pca, int rows ) {
    cv::FileStorage fs( out_path, cv::FileStorage::WRITE );
    fs << "pca" << pca;
    fs << "rows" << rows;
    fs.release();
}

static auto load_img_pca( const std::string & path ) {
    Image pca;
    int rows;
    cv::FileStorage fs( path, cv::FileStorage::READ );
    fs["pca"] >> pca ;
    fs["rows"] >> rows;
    fs.release();
    return std::make_tuple( pca, rows );
}

}

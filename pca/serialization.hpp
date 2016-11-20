#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <tuple>

namespace serial {

using Image = cv::Mat;

void save_pca(const std::string & file_name, cv::PCA pca ) {
    cv::FileStorage fs( file_name, cv::FileStorage::WRITE );
    fs << "mean" << pca.mean;
    fs << "e_vectors" << pca.eigenvectors;
    fs << "e_values" << pca.eigenvalues;
    fs.release();
}

cv::PCA load_pca( const std::string & file_name ) {
    cv::PCA pca;
    cv::FileStorage fs( file_name, cv::FileStorage::READ );
    fs["mean"] >> pca.mean ;
    fs["e_vectors"] >> pca.eigenvectors ;
    fs["e_values"] >> pca.eigenvalues ;
    fs.release();
    return pca;
}

void save_img_pca( const std::string & out_path, Image & pca, int rows ) {
    cv::FileStorage fs( out_path, cv::FileStorage::WRITE );
    fs << "pca" << pca;
    fs << "rows" << rows;
    fs.release();
}

auto load_img_pca( const std::string & path ) {
    Image pca;
    int rows;
    cv::FileStorage fs( path, cv::FileStorage::READ );
    fs["pca"] >> pca ;
    fs["rows"] >> rows;
    fs.release();
    return std::make_tuple( pca, rows );
}

}

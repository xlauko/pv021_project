#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "serialization.hpp"
#include "filter.hpp"

using Image = cv::Mat;

Image to_pca( Image & img, std::string pca_path ) {
    auto pca = serial::load_pca( pca_path );

    auto filtered = filter::filter( img );

    Image row = filtered.clone().reshape( 1, 1 );
    return pca.project( row );
}


Image from_pca( std::string path, std::string pca_path ) {
    auto img_pca = serial::load_img_pca( path );
    auto pca = serial::load_pca( pca_path );
    Image reconstructed = pca.backProject( std::get<0>( img_pca ) );
    return reconstructed.reshape( 1, std::get<1>( img_pca ) );
}

int main( int argc, char** argv )
{
    if( argc != 5 ) {
        std::cout <<" Usage: transform -{t/f} <img> <pca path> <out>" << std::endl;
        return -1;
    }

    std::string img_path = argv[2];
    std::string pca_path = argv[3];
    std::string out_path = argv[4];

    if ( std::string( argv[1] ) == "-t" ) {
        Image img = cv::imread( img_path, CV_LOAD_IMAGE_GRAYSCALE);
        auto transformed = to_pca( img, pca_path );
        serial::save_img_pca( out_path, transformed, img.rows );
    }

    if ( std::string( argv[1] ) == "-f" ) {
        auto image = from_pca( img_path, pca_path );
        cv::namedWindow( "PCA", cv::WINDOW_AUTOSIZE );
        cv::imshow( "PCA", image );

        cv::waitKey(0);
        cv::imwrite( out_path, image );
    }

    return 0;
}

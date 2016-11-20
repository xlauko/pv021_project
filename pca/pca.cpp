#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "serialization.hpp"
#include "filter.hpp"

using Image = cv::Mat;

Image reshape_images( std::vector< Image > & data ) {
    Image res( static_cast< int >( data.size() ), data[0].rows * data[0].cols, CV_8UC1 );

    size_t i = 0;
    for ( auto & img : data ) {
        Image row = img.clone().reshape( 1, 1 );
        row.convertTo( res.row( i ), CV_8UC1 );
        ++i;
    }

    return res;
}

cv::PCA compute_pca( std::vector< cv::String > filenames ) {
    std::vector< Image > imgs;

    for ( auto name : filenames ) {
        Image img = cv::imread( name, CV_LOAD_IMAGE_GRAYSCALE);

        if( !img.data ) {
            std::cout <<  "Could not open or find the image:" << name << std::endl ;
            std::exit( 1 );
        }

        auto filtered = filter::filter( img );
        imgs.push_back( filtered );
    }

    Image dataset = reshape_images( imgs );

    cv::PCA pca( dataset, Image(), 0, 0.95 );
    return pca;
}

int main( int argc, char** argv )
{
    if( argc != 3 ) {
        std::cout <<" Usage: pca <data set path> <out>" << std::endl;
        return -1;
    }

    std::vector< std::string > filenames;
    const std::string folder = argv[1];
    cv::glob( folder, filenames );

    auto pca = compute_pca( filenames );

    serial::save_pca( argv[2], pca );

    return 0;
}

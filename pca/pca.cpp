#include "pca.hpp"

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


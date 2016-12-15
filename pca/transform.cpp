#include "transform.hpp"
using Image = cv::Mat;

Image to_pca( Image & img, std::string pca_path ) {
    auto pca = serial::load_pca( pca_path );
    return to_pca( img, pca );
}

Image to_pca( Image & img, cv::PCA & pca ) {
    auto filtered = filter::filter( img );

    Image row = filtered.clone().reshape( 1, 1 );
    return pca.project( row );
}

Image from_pca( const std::string & path, const std::string & pca_path ) {
    auto pca = serial::load_pca( pca_path );
    return from_pca( path, pca );
}

Image from_pca( const std::string & path, cv::PCA & pca ) {
    auto img_pca = serial::load_img_pca( path );
    Image reconstructed = pca.backProject( std::get<0>( img_pca ) );
    return reconstructed.reshape( 1, std::get<1>( img_pca ) );
}

Image from_pca( Image & img_pca, size_t rows, cv::PCA & pca ) {
    Image reconstructed = pca.backProject( img_pca );
    return reconstructed.reshape( 1, rows );
}


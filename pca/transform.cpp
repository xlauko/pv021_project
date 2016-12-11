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

/*int main( int argc, char** argv )
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
}*/

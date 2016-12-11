#include <iostream>
#include <fstream>
#include <tuple>
#include <utility>
#include <iomanip>

#include "network/LstmCell.hpp"
#include "network/Network.hpp"
#include "network/util.hpp"

#include "pca/transform.hpp"

template < class Double = double >
struct NeuralFuns {
    using forgetAct = Sigmoid< Double >;
    using modulateAct = Tanh< Double >;
    using inputAct = Sigmoid< Double >;
    using outputAct = Sigmoid< Double >;
    using normalize = Tanh< Double >;
};

constexpr size_t input_size = 17; //pca.eigenvalues.rows;
constexpr size_t batch_size = 10; //pca.eigenvalues.rows;
using Cell = LstmCell< input_size, input_size, NeuralFuns<> >;
using Net = Network< double, Cell, Cell, Cell, Cell, Cell >;

using Input = std::array< double, input_size >;
using Output = std::array< double, input_size >;

Input to_array( Image & pca ) {
    std::vector< double > v;
    pca.row(0).copyTo( v );
    Input arr;
    std::copy_n(std::make_move_iterator(v.begin()), input_size, arr.begin());
    return arr;
}

template< typename PCA, typename Network >
void learn( PCA& pca, Network& n, std::vector< std::string >& paths ) {
    double scale = 1;
    size_t rows = 1;

    std::cout << "Loading input..." << std::endl;
    std::vector< Image > loaded;
    for ( auto & path : paths ) {
        Image img = cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE);
        rows = img.rows;
        auto transformed = to_pca( img, pca );
        assert( transformed.cols == input_size );

        double max, min;
        cv::minMaxLoc( transformed, &min, &max );
        max = std::fabs( min ) > max ? std::fabs( min ) : max;
        scale = scale < max ? max : scale;

        loaded.push_back( transformed );
    }

    std::vector< Input > imgs;
    //std::cout << loaded[ 0 ] << std::endl;
    for ( auto & img : loaded ) {
        img = img / scale;
        imgs.push_back( to_array( img ) );
    }

    std::cout << "Learning..." << std::endl;
    n.randomizeWeights(-1, 1);
    for ( int k = 0; k < 1000; ++k )
    for ( int i = 0; i < imgs.size() - batch_size - 1; ++i ) {
        ArrayView< Input, batch_size > batch = &imgs[ i ];
        Input ex = imgs[ i + batch_size ];
        n.learn( { batch.begin(), batch.end() }, ex, 0.5 );
    }

    ArrayView< Input, batch_size > test = &imgs[ 0 ];
    n.evaluate( { test.begin(), test.end() } );
    //std::cout << n.output << std::endl;
    //std::cout << imgs[ batch_size ] << std::endl;

    //auto img_pca = cv::Mat( { n.output.begin(), n.output.end() }, true );
    auto img_pca = cv::Mat( 1, input_size, CV_64F, n.output.begin() );
    img_pca = img_pca * scale;
    //std::cout << img_pca << std::endl;
    auto image = from_pca( img_pca, rows, pca );
    cv::namedWindow( "PCA", cv::WINDOW_AUTOSIZE );
    cv::imshow( "PCA", image );

    cv::imwrite( "forecast.jpg", image );
    cv::waitKey(0);
}



int main() {
    const std::string path = "test.pca";
    const std::string in = "data"; //argv[1];

    std::vector< std::string > filenames;
    cv::glob( in, filenames );

    std::cout << "Loading pca..." <<std::endl;
    auto pca = serial::load_pca( path );
    assert( pca.eigenvalues.rows == input_size );

    Net network;
    learn< decltype(pca), Net >( pca, network, filenames );

    return 0;
}

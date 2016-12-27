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

constexpr size_t input_size = 150; //pca.eigenvalues.rows;
constexpr size_t batch_size = 7;
using Cell = LstmCell< input_size, input_size, NeuralFuns<> >;
using Net = Network< double, Cell, Cell >;

using Input = std::array< double, input_size >;
using Output = std::array< double, input_size >;

double scale = 1;
size_t rows = 1;

Input to_array( Image & pca ) {
    std::vector< double > v;
    pca.row(0).copyTo( v );
    Input arr;
    std::copy_n(std::make_move_iterator(v.begin()), input_size, arr.begin());
    return arr;
}

template< typename PCA >
std::vector< Input > loadInput( std::vector< cv::String >& paths, PCA& pca ) {
    std::cout << "Loading input..." << std::endl;
    std::vector< Image > loaded;
    for ( const auto & path : paths ) {
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
    for ( auto & img : loaded ) {
        img = img / scale;
        imgs.push_back( to_array( img ) );
    }

    return imgs;
}

template< typename PCA, typename Network >
void learn( PCA& pca, Network& n, std::vector< cv::String >& paths, int ite ) {
    auto imgs = loadInput< PCA >( paths, pca );
    std::cout << "Learning..." << std::endl;
    for ( int k = 0; k < ite; ++k ) {
        std::cout << "Learning round: " << k << std::endl;
        for ( int i = 0; i < imgs.size() - batch_size - 1; ++i ) {
            ArrayView< Input, batch_size > batch = &imgs[ i ];
            Input ex = imgs[ i + batch_size ];
            n.learn( { batch.begin(), batch.end() }, ex, 1 );
        }
    }
}

template< typename PCA, typename Network >
void evaluate( PCA& pca, Network& n, std::vector< cv::String >& paths ) {
    auto imgs = loadInput< PCA >( paths, pca );
    std::cout << "Evaluating..." << std::endl;

    ArrayView< Input, batch_size > test = &imgs[ 0 ];
    n.evaluate( { test.begin(), test.end() } );
}

int main( int argc, const char* argv[] )
{
    std::string keys = {
        "{ data | | data path }"
        "{ pca     | | pca path }"
        "{ learn   |1| learn if 1 else evaluate }"
        "{ o      |<none>| evaluated output }"
        "{ @l      |<none>| load network from given path }"
        "{ @s      | | save network to given path }"
        "{ ite    || number of learning iterations }"
    };

    cv::CommandLineParser cmd( argc, argv, keys.c_str() );

    if( argc < 3 )
    {
        //cmd.printMessage();
        return 0;
    }

    const std::string path = cmd.get<std::string>("pca");
    const std::string in = cmd.get<std::string>("data");

    std::vector< cv::String > filenames;
    cv::glob( in, filenames );

    std::cout << "Loading PCA..." << path << std::endl;
    auto pca = serial::load_pca( path );
    std::cout << "PCA size: " << pca.eigenvalues.rows << std::endl;
    assert( pca.eigenvalues.rows == input_size
        && "Change input_size in main.cpp to correct pca size." );

    Net network;

    auto l = cmd.get< std::string >( "l" );
    if ( l != "" ) {
        std::ifstream fi( l );
        read( fi, network );
    } else {
        network.randomizeWeights(-1, 1);
    }

    bool tolearn = cmd.get< int >( "learn" );
    if ( tolearn ) {
        int ite = cmd.get< int >( "ite" );
        learn< decltype(pca), Net >( pca, network, filenames, ite );
    } else {
        evaluate< decltype(pca), Net >( pca, network, filenames );
        auto img_pca = cv::Mat( 1, input_size, CV_64F, network.output.begin() );
        img_pca = img_pca * scale;
        auto image = from_pca( img_pca, rows, pca );
        auto o = cmd.get< std::string >( "o" );
        if ( o == "" )
            std::cerr << "Empty output path." << std::endl;
        cv::imwrite( o, image );
    }

    auto s = cmd.get< std::string >( "s" );
    if ( s != "" ) {
        std::ofstream fo( s );
        write( fo, network );
    }

    return 0;
}

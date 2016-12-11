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
using Net = Network< double, Cell, Cell >;

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
    std::vector< Input > imgs;
    double scale = 1;

    std::cout << "Loading input..." << std::endl;
    std::vector< Image > loaded;
    for ( auto & path : paths ) {
        Image img = cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE);
        auto transformed = to_pca( img, pca );
        assert( transformed.cols == input_size );

        double max, min;
        cv::minMaxLoc( transformed, &min, &max );
        max = std::fabs( min ) > max ? std::fabs( min ) : max;
        scale = scale < max ? max : scale;

        loaded.push_back( transformed );
    }

    for ( auto & img : loaded ) {
        img = img / scale;
        imgs.push_back( to_array( img ) );
    }

    std::cout << "Learning..." << std::endl;
    n.randomizeWeights(-1, 1);
    for ( int i = 0; i < imgs.size() - batch_size - 1; ++i ) {
        ArrayView< Input, batch_size > batch = &imgs[ i ];
        Input ex = imgs[ i + batch_size ];
        n.learn( { batch.begin(), batch.end() }, ex, 1 );
    }

    ArrayView< Input, batch_size > test = &imgs[ 0 ];
    n.evaluate( { test.begin(), test.end() } );
    std::cout << n.output << std::endl;
}



int main() {
    /*std::cout << std::fixed << std::setprecision(5);
    using MyCell = LstmCell< 2, 2, NeuralFuns<> >;
    using Net = Network< double, MyCell, MyCell, MyCell >;

    std::vector< std::array< double, 2 > > in;
    for ( int i = 0; i != 2; i++ ) {
        in.push_back( {1, 2} );
    }

    Net n;
    n.randomizeWeights( -1, 1 );
    for ( int i = 0; i != 1000; i++ ) {
        std::array< double, 2 > o{ 1, 0 };
        n.learn( in, o, 0.1);
    }

    n.evaluate( in );
    for ( auto x : n.output )
        std::cout << x << ", ";

    return 0;
*/

    const std::string path = "test.pca";
    const std::string in = "data"; //argv[1];

    std::vector< std::string > filenames;
    cv::glob( in, filenames );

    std::cout << "Loading pca..." <<std::endl;
    auto pca = serial::load_pca( path );
    assert( pca.eigenvalues.rows == input_size );

    Net network;
    learn< decltype(pca), Net >( pca, network, filenames );

    /*std::tuple< int, int, int > tup( 42, 43, 44 );

    LstmCell< 10, 20, NeuralFuns<> > cell;
    cell.randomizeWeights( -1, 1 );
    cell.forwardPropagate();
    std::ofstream fo( "test.txt" );
    write( fo, cell );
    fo.close();
    std::ifstream fi( "test.txt" );
    read( fi, cell );
    std::cout << "Here should be output of the network. But there is none\n";

    std::array< double, 20 > t;
    LstmCell< 10, 20, NeuralFuns<> >::LearningContext c;
    cell.backPropagate( t.data(), t.data(), c );

    std::cout << "Sizes: " << cell.inputSize << ", " << cell.outputSize << "\n";

    Network< double, MyCell, MyCell, MyCell > network;
    network.forwardPropagate();
    network.evaluate( {} );
    decltype(network)::Output ex;
    network.learn( {}, ex, 0 );*/
    return 0;
}

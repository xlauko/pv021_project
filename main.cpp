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
using Cell = LstmCell< input_size, input_size, NeuralFuns<> >;
using LSTMNetwork = Network< double, Cell, Cell, Cell >;
using Input = std::array< double, input_size >;

Input to_array( Image & pca ) {
    std::vector< double > v;
    pca.row(0).copyTo( v );
    Input arr;
    std::copy_n(std::make_move_iterator(v.begin()), input_size, arr.begin());
    return arr;
}

template< typename PCA, typename Network >
void learn( PCA& pca, Network& network, std::vector< std::string >& paths, std::string exp ) {
    std::vector< Input > imgs;
    std::cout << "Loading input..." << std::endl;
    for ( auto & path : paths ) {
        Image img = cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE);
        auto transformed = to_pca( img, pca );
        assert( transformed.cols == input_size );
        imgs.push_back( to_array( transformed ) );
    }
    std::cout << "Learning..." << std::endl;
    Image exp_img = cv::imread( exp, CV_LOAD_IMAGE_GRAYSCALE);
    auto exp_out = to_pca( exp_img, pca );

    LSTMNetwork::Output ex = to_array( exp_out );
    /*network.learn( imgs, ex, 10 );
    for ( auto a: output )
        std::cout << a << std::endl;
*/
    network.evaluate( imgs );
    for ( auto a: network.output )
        std::cout << a << std::endl;
}



int main() {
    std::cout << std::fixed << std::setprecision(5);
    using MyCell = LstmCell< 2, 2, NeuralFuns<> >;
    using Net = Network< double, MyCell, MyCell, MyCell >;

    std::vector< std::array< double, 2 > > in;
    for ( int i = 0; i != 2; i++ ) {
        in.push_back( {1, 2} );
    }

    Net n;
    n.randomizeWeights( 0, 1 );
    for ( int i = 0; i != 1; i++ ) {
        std::array< double, 2 > o{ 1, 0 };
        n.learn( in, o, 0.1);
    }

    /*n.evaluate( in );
    for ( auto x : n.output )
        std::cout << x << ", ";*/

    return 0;


    const std::string path = "test.pca";
    const std::string img = "2015-12-24-23-45.jpg";

    std::cout << "Loading pca..." <<std::endl;
    auto pca = serial::load_pca( path );
    assert( pca.eigenvalues.rows == input_size );

    LSTMNetwork network;
    std::vector< std::string > paths = { img };
    learn< decltype(pca), LSTMNetwork >( pca, network, paths, img );

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

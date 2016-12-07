#include <iostream>
#include <fstream>

#include "LstmCell.hpp"
#include "Network.hpp"
#include "util.hpp"

template < class Double = double >
struct NeuralFuns {
    using forgetAct = Sigmoid< Double >;
    using modulateAct = Tanh< Double >;
    using inputAct = Sigmoid< Double >;
    using outputAct = Sigmoid< Double >;
    using normalize = Tanh< Double >;
};

using MyCell = LstmCell< 20, 20, NeuralFuns<> >;

int main() {
    std::tuple< int, int, int > tup( 42, 43, 44 );
    for_each_in_tuple( tup, [](auto x) { std::cout << "F: " << x << "\n"; }, false );
    for_each_in_tuple( tup, [](auto x) { std::cout << "B: " << x << "\n"; }, true );

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
    cell.backwardPropagate( t.data(), t.data(), c );

    std::cout << "Sizes: " << cell.inputSize << ", " << cell.outputSize << "\n";

    Network< double, MyCell, MyCell, MyCell> network;
    return 0;
}

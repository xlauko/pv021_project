#include <iostream>
#include <fstream>

#include "LstmCell.hpp"

template < class Double = double >
struct NeuralFuns {
    using forgetAct = Sigmoid< Double >;
    using modulateAct = Tanh< Double >;
    using inputAct = Sigmoid< Double >;
    using outputAct = Sigmoid< Double >;
    using normalize = Tanh< Double >;
};

int main() {
    LstmCell< 10, 20, NeuralFuns<> > cell;
    cell.randomizeWeights( -1, 1 );
    cell.forwardPropagate();
    std::ofstream fo( "test.txt" );
    write( fo, cell );
    fo.close();
    std::ifstream fi( "test.txt" );
    read( fi, cell );
    std::cout << "Here should be output of the network. But there is none\n";
    return 0;
}

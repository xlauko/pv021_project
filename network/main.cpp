#include <iostream>
#include <fstream>

#include "LstmCell.hpp"

template < class Double = double >
struct NeuralFuns {
    using forgetAct = Sigmoid< Double >;
    using modulateAct = Tanh< Double >;
    using inputAct = Sigmoid< Double >;
    using outputAct = Sigmoid< Double >;
    static Double normalize( Double x ) {
        return tanh( x );
    }
};

int main() {
    LstmCell< 10, 20, NeuralFuns<> > cell;
    cell.forwardPropagate();
    std::ofstream fo( "test.txt" );
    write( fo, cell._forgetGate );
    fo.close();
    std::ifstream fi( "test.txt" );
    read( fi, cell._forgetGate );
    std::cout << "Here should be output of the network. But there is none\n";
    return 0;
}

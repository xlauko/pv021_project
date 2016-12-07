#pragma once

#include <algorithm>
#include "LstmCell.hpp"

template < int InputSize, int InterSize, int OutputSize, class Funs, class Double = double >
struct SimpleNetwork {

    void forwardPropagate() {
        _inputLayer.forwardPropagate();
        std::copy( _inputLayer._output.begin(), _inputLayer._output.end(),
            _output._input.begin() );
        _outputLayer.forwardPropagate();
    }

    LstmCell< InputSize, InterSize, Funs, Double > _inputLayer;
    LstmCell< InterSize, OutputSize, Funs, Double > _outputLayer;
}

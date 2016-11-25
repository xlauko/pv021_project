#pragma once

#include <array>
#include <cmath>
#include <functional>
#include "NeuralLayer.hpp"

template < class I1, class I2, class O, class Op >
constexpr auto biElementWise = std::transform< I1, I2, O, Op >;

template < class I1, class I2, class I3, class O, class Op >
O triElementWise( I1 beg1, I1 end1, I2 beg2, I3 beg3, O out, Op op ) {
    while ( beg1 != end1 )
        *out++ = op( *beg1++, *beg2++, *beg3++ );
    return out;
}

template < int InputSize, int OutputSize, class Funs, class Double = double >
struct LstmCell {
    LstmCell() :
        _forgetGate( _concatInput.data() ),
        _modulateGate( _concatInput.data() ),
        _inputGate( _concatInput.data() ),
        _outputGate( _concatInput.data() )
    {}

    void forwardPropagate() {
        _forgetGate.forwardPropagate();
        _modulateGate.forwardPropagate();
        _inputGate.forwardPropagate();
        _outputGate.forwardPropagate();

        std::transform( _memory.begin(), _memory.end(),
            _forgetGate.output(), _memory.begin(),
            std::multiplies< Double >() );
        triElementWise( _memory.begin(), _memory.end(),
            _modulateGate.output(), _inputGate.output(), _memory.begin(),
            []( Double mem, Double a, Double b ) {
                return mem + a * b;
            });
        std::transform( _memory.begin(), _memory.end(),
            _outputGate.output(), _memory.begin(),
            []( Double mem, Double e ) {
                return Funs::normalize( mem ) * e;
            });
    }

    Double *input() { return _concatInput.data(); }
    Double *output() { return _concatInput.data() + InputSize; }

    std::array< Double, InputSize + OutputSize > _concatInput;
    std::array< Double, OutputSize > _memory;
    std::array< Double, OutputSize > _output;

    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::forgetAct, Double > _forgetGate;
    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::modulateAct, Double > _modulateGate;
    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::inputAct, Double > _inputGate;
    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::outputAct, Double > _outputGate;
};
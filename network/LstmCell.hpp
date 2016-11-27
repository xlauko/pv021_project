#pragma once

#include <array>
#include <cmath>
#include <functional>
#include "ArrayView.hpp"
#include "NeuralLayer.hpp"

template < class I1, class I2, class O, class Op >
O biElementWise( I1&& beg1, I1&& end1, I2&& beg2, O&& out, Op&& op ) {
    return std::transform( std::forward< I1 >( beg1 ), std::forward< I1 >( end1 ),
        std::forward< I2 >( beg2 ), std::forward< O >( out ),
        std::forward < Op>( op ) );
}

template < class I1, class I2, class I3, class O, class Op >
O triElementWise( I1 beg1, I1 end1, I2 beg2, I3 beg3, O out, Op op ) {
    while ( beg1 != end1 )
        *out++ = op( *beg1++, *beg2++, *beg3++ );
    return out;
}

template < int InputSize, int OutputSize, class Funs, class Double = double >
struct LstmCell {
    LstmCell() :
        _output( _concatInput.data() + InputSize ),
        _input( _concatInput.data() ),
        _forgetGate( _concatInput ),
        _modulateGate( _concatInput ),
        _inputGate( _concatInput ),
        _outputGate( _concatInput )
    {}

    void forwardPropagate() {
        _forgetGate.forwardPropagate();
        _modulateGate.forwardPropagate();
        _inputGate.forwardPropagate();
        _outputGate.forwardPropagate();

        biElementWise( _memory.begin(), _memory.end(),
            _forgetGate._output.begin(), _memory.begin(),
            std::multiplies< Double >() );
        triElementWise( _memory.begin(), _memory.end(),
            _modulateGate._output.begin(), _inputGate._output.begin(),
            _memory.begin(),
            []( Double mem, Double a, Double b ) {
                return mem + a * b;
            });
        biElementWise( _memory.begin(), _memory.end(),
            _outputGate._output.begin(), _memory.begin(),
            []( Double mem, Double e ) {
                return Funs::normalize( mem ) * e;
            });
    }

    void randomizeMemory( Double min, Double max ) {
        _randomize( min, max, _memory );
    }

    void randomizeInput( Double min, Double max ) {
        _randomize( min, max, _concatInput );
    }

    template < class T >
    void _randomize( Double min, Double max, T& vec ) {
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_real_distribution<> distribution(min, max);
        auto rgen = std::bind( distribution, generator );
        std::generate( vec.begin(), vec.end(), rgen );
    }

    std::array< Double, InputSize + OutputSize > _concatInput;
    std::array< Double, OutputSize > _memory;
    ArrayView< Double, OutputSize > _output;
    ArrayView< Double, InputSize > _input;

    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::forgetAct, Double > _forgetGate;
    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::modulateAct, Double > _modulateGate;
    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::inputAct, Double > _inputGate;
    NeuralLayer< InputSize + OutputSize, OutputSize,
        typename Funs::outputAct, Double > _outputGate;
};

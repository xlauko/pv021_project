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
        std::forward < Op >( op ) );
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

    LstmCell( const LstmCell& o ) :
        _concatInput( o._concatInput ),
        _memory( o._memory ),
        _output( _concatInput.data() + InputSize ),
        _input( _concatInput.data() ),
        _forgetGate( _concatInput ),
        _modulateGate( _concatInput ),
        _inputGate( _concatInput ),
        _outputGate( _concatInput )
    {
        _forgetGate._copyState( o._forgetGate );
        _modulateGate._copyState( o._modulateGate );
        _inputGate._copyState( o._inputGate );
        _outputGate._copyState( o._outputGate );
    }

    LstmCell& operator=( const LstmCell& o ) {
        if ( this == &o )
            return *this;
        _forgetGate = o._forgetGate;
        _modulateGate = o._modulateGate;
        _inputGate = o._modulateGate;
        _outputGate = o._modulateGate;

        _concatInput = o._concatInput;
        _memory = o._memory;
    }

    void forwardPropagate() {
        _forgetGate.forwardPropagate();
        _modulateGate.forwardPropagate();
        _inputGate.forwardPropagate();
        _outputGate.forwardPropagate();

        biElementWise( _memory.begin(), _memory.end(),
            _forgetGate._output.begin(),
            _memory.begin(),
            std::multiplies< Double >() );
        triElementWise( _memory.begin(), _memory.end(),
            _modulateGate._output.begin(), _inputGate._output.begin(),
            _memory.begin(),
            []( Double mem, Double a, Double b ) {
                return mem + a * b;
            });
        biElementWise( _memory.begin(), _memory.end(),
            _outputGate._output.begin(),
            _output.begin(),
            []( Double mem, Double e ) {
                return Funs::normalize::f( mem ) * e;
            });
    }

    // Perfom back propagation, return memory gradient
    std::array< Double, OutputSize > backwardPropagate(
        ArrayView< Double, OutputSize > desiredOutput,
        ArrayView< Double, OutputSize > prevMemory,
        ArrayView< Double, OutputSize > memoryGradient )
    {
        // Follow: http://arunmallya.github.io/writeups/nn/lstm/index.html
        std::array< Double, OutputSize > dH;
        biElementWise( _output.begin(), _output.end(), desiredOutput.begin(),
            dH.begin(),
            []( Double out, Double ex) {
                return 2 * ( out - ex );
            } );

        std::array< Double, OutputSize > dO;
        biElementWise( dH.begin(), dH.end(), _memory.begin(), dO.begin(),
            []( Double dh, Double c ) {
                return dh * Funs::normalize::f( c );
            } );

        std::array< Double, OutputSize > dC;
        triElementWise( dH.begin(), dH.end(), _outputGate._output.begin(),
            _memory.begin(), dC.begin(),
            []( Double dh, Double o, Double c ) {
                return dh * o * Funs::normalize::d( c );
            } );
        biElementWise( dC.begin(), dC.end(), memoryGradient.begin(), dC.begin(),
            std::plus< Double >() );

        std::array< Double, OutputSize > dI;
        biElementWise( dC.begin(), dC.end(), _modulateGate.begin(), dI.begin(),
            std::multiplies< Double >() );

        std::array< Double, OutputSize > dF;
        biElementWise( dC.begin(), dC.end(), prevMemory.begin(), dF.begin(),
            std::multiplies< Double >() );

        std::array< Double, OutputSize > dA;
        biElementWise( dC.begin(), dC.end(), _inputGate._output.begin(),
            std::multiplies< Double >() );

        std::array< Double, OutputSize > dC1;
        biElementWise( dC.begin(), dC.end(), _forgetGate._output.begin(),
            dC1.begin(), std::multiplies< Double >() );

        std::array< Double, OutputSize > dAp;
        biElementWise( dA.begin(), dA.end(), _modulateGate._output.begin(),
            dAp.begin(),
            []( Double da, Double a ) {
                return da * Funs::modulateAct::df( a );
            } );

        std::array< Double, OutputSize > dIp;
        biElementWise( dI.begin(), dI.end(), _inputGate._output.begin(),
            dIp.begin(),
            []( Double di, Double i ) {
                return di * i * ( Double( 1.0 ) - i );
            } );

        std::array< Double, OutputSize > dFp;
        biElementWise( dF.begin(), dF.end(), _forgetGate._output.begin(),
            dFp.begin(),
            []( Double df, Double f) {
                return df * f * ( Double( 1.0 ) - f );
            } );

        std::array< Double, OutputSize > dOp;
        biElementWise( dO.begin(), dO.end(), _outputGate._output.begin(),
            dIp.begin(),
            []( Double d, Double o ) {
                return d * o * ( Double( 1.0 ) - o );
            } );

        return dC;
    }

    void randomizeWeights( Double min, Double max ) {
        _forgetGate.randomizeWeights( min, max );
        _modulateGate.randomizeWeights( min, max );
        _inputGate.randomizeWeights( min, max );
        _outputGate.randomizeWeights( min, max );
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

template < int InputSize, int OutputSize, class Funs, class Double >
void write( std::ostream& s, const LstmCell< InputSize, OutputSize, Funs, Double>& c ) {
    s.write( "LC", 2 );
    int tmp = InputSize;
    const char *data = reinterpret_cast< const char * >( &tmp );
    s.write( data, sizeof( InputSize ) );
    tmp = OutputSize;
    s.write( data, sizeof( OutputSize ) );
    tmp = sizeof( Double );
    s.write( data, sizeof( tmp ) );

    write( s, c._forgetGate );
    write( s, c._modulateGate );
    write( s, c._inputGate );
    write( s, c._outputGate );
}

template < int InputSize, int OutputSize, class Funs, class Double >
void read( std::istream& s, LstmCell< InputSize, OutputSize, Funs, Double>& c ) {
    char type[ 2 ];
    s.read( type, 2 );
    if ( type[ 0 ] != 'L' || type[ 1 ] != 'C' )
        throw std::runtime_error( "Unexpected type" );
    int tmp;
    char *data = reinterpret_cast< char * >( &tmp );
    s.read( data, sizeof( tmp ) );
    if ( tmp != InputSize )
        throw std::runtime_error( "Unexpected InputSize of a LstmCell" );
    s.read( data, sizeof( tmp ) );
    if ( tmp != OutputSize )
        throw std::runtime_error( "Unexpected OutputSize of a LstmCell" );
    s.read( data, sizeof( tmp ) );
    if ( tmp != sizeof( Double ) )
        throw std::runtime_error( "Unexpected size of Double of a LstmCell" );

    read( s, c._forgetGate );
    read( s, c._modulateGate );
    read( s, c._inputGate );
    read( s, c._outputGate );
}

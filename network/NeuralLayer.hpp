#pragma once

#include <cassert>
#include <array>
#include <algorithm>

template < class Double = double >
struct Tanh {
    static Double f( Double x ) {
        return tanh( x );
    }
    static Double d( Double x ) {
        Double t = tanh( x );
        return Double( 1.0 ) - t * t;
    }
};

template < class Double = double >
struct Sigmoid {
    static Double f( Double x ) {
        return Double( 1.0 ) / ( Double( 1.0 ) + exp( -x ) );
    }
    static Double d( Double x ) {
        Double t = Double( 1.0 ) + expr( x );
        return exp( x ) / ( t * t );
    }
};


template < int InputSize, int OutputSize, class Fun, class Double = double >
struct NeuralLayer {
    NeuralLayer( Double *input = nullptr ) : _input( input ) { }

    void forwardPropagate() {
        assert( _input );
        for ( int i = 0; i != OutputSize; i++ ) {
            Double prod = std::inner_product( _weights[ i ].begin() + 1,
                _weights[ i ].end(), _input, _weights[ i ][ 0 ] );
            _output[ i ] = Fun::f( prod );
        }
    }

    Double *input() { return _input; }
    Double *output() { return _output.data(); }

    Double *_input;
    std::array< Double, OutputSize > _output;
    std::array< std::array< Double, InputSize + 1 >, OutputSize > _weights;
};

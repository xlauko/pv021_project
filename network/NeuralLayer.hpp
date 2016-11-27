#pragma once

#include <cassert>
#include <array>
#include <algorithm>
#include <iostream>
#include <exception>
#include "ArrayView.hpp"

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
    NeuralLayer( ArrayView< Double, InputSize > input ) : _input( input ) {};
    NeuralLayer( std::array< Double, InputSize >& input )
        : _input( input.data() )
    {};

    void forwardPropagate() {
        assert( _input );
        for ( int i = 0; i != OutputSize; i++ ) {
            Double prod = std::inner_product( _weights[ i ].begin() + 1,
                _weights[ i ].end(), _input.begin(), _weights[ i ][ 0 ] );
            _output[ i ] = Fun::f( prod );
        }
    }

    void randomizeWeights( Double min, Double max ) {
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_real_distribution<> distribution(min, max);
        auto rgen = std::bind( distribution, generator );
        for ( auto& vec : _weights )
            std::generate( vec.begin(), vec.end(), rgen );
    }

    ArrayView< Double, InputSize > _input;
    std::array< Double, OutputSize > _output;
    std::array< std::array< Double, InputSize + 1 >, OutputSize > _weights;
};

template < int InputSize, int OutputSize, class Fun, class Double >
void write( std::ostream& s, const NeuralLayer< InputSize, OutputSize, Fun, Double >& l ) {
    s.write( "NL", 2 );
    int tmp = InputSize;
    const char *data = reinterpret_cast< const char * >( &tmp );
    s.write( data, sizeof( InputSize ) );
    tmp = OutputSize;
    s.write( data, sizeof( OutputSize ) );
    tmp = sizeof( Double );
    s.write( data, sizeof( tmp ) );

    for ( const auto& w : l._weights )
        s.write( reinterpret_cast< const char * >( w.data() ),
            sizeof( Double ) * (InputSize + 1) );
}

template < int InputSize, int OutputSize, class Fun, class Double >
void read( std::istream& s, NeuralLayer< InputSize, OutputSize, Fun, Double >& l ) {
    char type[ 2 ];
    s.read( type, 2 );
    if ( type[ 0 ] != 'N' || type[ 1 ] != 'L' )
        throw std::runtime_error( "Unexpected type" );
    int tmp;
    char *data = reinterpret_cast< char * >( &tmp );
    s.read( data, sizeof( tmp ) );
    if ( tmp != InputSize )
        throw std::runtime_error( "Unexpected InputSize of a NL" );
    s.read( data, sizeof( tmp ) );
    if ( tmp != OutputSize )
        throw std::runtime_error( "Unexpected OutputSize of a NL" );
    s.read( data, sizeof( tmp ) );
    if ( tmp != sizeof( Double ) )
        throw std::runtime_error( "Unexpected size of Double of a NL" );

    for ( auto& w : l._weights )
        s.read( reinterpret_cast< char * >( w.data() ),
            sizeof( Double ) * (InputSize + 1) );
}

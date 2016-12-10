#pragma once

#include <cassert>
#include <array>
#include <random>
#include <algorithm>
#include <iostream>
#include <exception>
#include "ArrayView.hpp"

template < class Double = double >
struct Tanh {
    // function value
    static Double f( Double x ) {
        return tanh( x );
    }
    // derivative
    static Double d( Double x ) {
        Double t = tanh( x );
        return Double( 1.0 ) - t * t;
    }
    // derivative based on function value
    static Double df( Double x ) {
        return Double( 1.0 ) - x * x;
    }
};

template < class Double = double >
struct Sigmoid {
    // function value
    static Double f( Double x ) {
        return Double( 1.0 ) / ( Double( 1.0 ) + exp( -x ) );
    }
    // derivative
    static Double d( Double x ) {
        Double t = Double( 1.0 ) + exp( x );
        return exp( x ) / ( t * t );
    }
    // deriviative based on function value
    static Double df( Double x ) {
        Double t = Double( 1.0 ) + x;
        return x / ( t * t );
    }
};


template < int InputSize, int OutputSize, class Fun, class Double = double >
struct NeuralLayer {
    NeuralLayer( ArrayView< Double, InputSize > input ) : _input( input ) {};
    NeuralLayer( std::array< Double, InputSize >& input )
        : _input( input.data() )
    {
        for ( auto& x : _weights )
            std::fill( x.begin(), x.end(), Double( 0.0 ) );
        clear();
    };

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

    void _copyState( const NeuralLayer& l ) {
        _output = l._output;
        _weights = l._weights;
    }

    void clear() {
        std::fill( _output.begin(), _output.end(), 0.0 );
    }

    void adjustWeights( std::array< std::array< Double, InputSize + 1 >, OutputSize >& d,
        Double step )
    {
        for( int i = 0; i != OutputSize; i++ ) {
            for ( int j = 0; j != InputSize + 1; j++ )
                _weights[ i ][ j ] += d[ i ][ j ] * step;
        }
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

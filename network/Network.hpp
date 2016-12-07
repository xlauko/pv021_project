#pragma once
#include <type_traits>
#include "ArrayView.hpp"
#include "util.hpp"

template < class Double, class... Cells /*bottom-up*/ >
struct Network {
    static constexpr int _layerCount = sizeof...( Cells );
    using Layers = std::tuple< Cells... >;

    Layers _layers;

    using InputLayerType =
        typename std::remove_reference< decltype( std::get< 0 >( _layers ) ) >::type;
    using OutputLayerType =
        typename std::remove_reference< decltype( std::get< _layerCount - 1 >( _layers ) ) >::type;
    ArrayView< Double, InputLayerType::inputSize > input;
    ArrayView< Double, OutputLayerType::outputSize > output;

    using Input = std::array< Double, InputLayerType::inputSize >;
    using Output = std::array< Double, OutputLayerType::inputSize >;

    Network( )
        : input( std::get< 0 >( _layers )._input ),
          output( std::get< _layerCount - 1 >( _layers )._output )
    {}

    void forwardPropagate() {
        _forwardPropagate( _layers, std::integral_constant< int, _layerCount>() );
    }

    void forwardPropagate( Layers& l ) {
        _forwardPropagate( l, std::integral_constant< int, _layerCount>() );
    }

    void _forwardPropagate( Layers& l, std::integral_constant< int, 1 > ) {
        const int idx = _layerCount - 1;
        std::get< idx >( l ).forwardPropagate();
    }

    template < class I >
    void _forwardPropagate( Layers& l, I ) {
        const int idx = _layerCount - I::value;
        auto& source = std::get< idx >( l );
        auto& target = std::get< idx + 1 >( l );

        using A = typename std::remove_reference< decltype( source ) >::type;
        using B = typename std::remove_reference< decltype( target ) >::type;
        static_assert( A::outputSize == B::inputSize );

        source.forwardPropagate();
        std::copy( source._output.begin(), source._output.end(),
            target._input.begin() );

        _forwardPropagate( l, std::integral_constant< int, I::value - 1 >() );
    }

    void evaluate( std::vector< Input > in ) {
        for ( const auto& frame : in ) {
            std::copy( frame.begin(), frame.end(), input.begin() );
            forwardPropagate();
        }
    }

    void learn( std::vector< std::pair< Input, Output> > sample ) {
        std::vector< Layers > timeSteps;
        Layers *prev = &_layers;
        // Unroll forward propagation in time
        for ( const auto& frame : sample ) {
            timeSteps.push_back( *prev );
            prev = &timeSteps.back();
            auto& initial = std::get< 0 >( *prev );
            std::copy( frame.first.begin(), frame.first.end(), initial._input.begin() );
            forwardPropagate( *prev );
        }
        // Back propage the desired output
    }


};

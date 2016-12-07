#pragma once
#include "ArrayView.hpp"
#include "util.hpp"

template < class Double, class InputCell, class OutputCell, class... Cells /*bottom-up*/ >
struct Network {
    static constexpr int _layerCount = sizeof...( Cells ) + 2;

    Network( )
        : _input( std::get< 0 >( _layers )._input ),
          _output( std::get< _layerCount - 1 >( _layers )._output )
    {}

    void forwardPropagate() {
        _forwardPropagate( std::integral_constant< int, _layerCount>() );
    }

    void _forwardPropagate( std::integral_constant< int, 1 > ) {
        const int idx = _layerCount - 1;
        std::get< idx >( _layers ).forwardPropagate();
    }

    template < class I >
    void _forwardPropagate( I ) {
        const int idx = _layerCount - I::value;
        auto& source = std::get< idx >( _layers );
        auto& target = std::get< idx + 1 >( _layers );

        source.forwardPropagate();
        std::copy( source._output.begin(), source._output.end(),
            target._input.begin() );

        _forwardPropagate( std::integral_constant< int, I::value - 1>() );
    }

    std::tuple< InputCell, Cells..., OutputCell > _layers;
    ArrayView< Double, InputCell::inputSize > _input;
    ArrayView< Double, OutputCell::outputSize > _output;
};

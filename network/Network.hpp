#pragma once
#include <type_traits>
#include "LstmCell.hpp"
#include "ArrayView.hpp"
#include "util.hpp"

template < class Double, class... Cells /*bottom-up*/ >
struct Network {
    static constexpr int _layerCount = sizeof...( Cells );
    using Layers = std::tuple< Cells... >;
    template< typename T > using GetContext = typename T::LearningContext;

    Layers _layers;

    using InputLayerType =
        typename std::remove_reference< decltype( std::get< 0 >( _layers ) ) >::type;
    using OutputLayerType =
        typename std::remove_reference< decltype( std::get< _layerCount - 1 >( _layers ) ) >::type;
    ArrayView< Double, InputLayerType::inputSize > input;
    ArrayView< Double, OutputLayerType::outputSize > output;

    using Input = std::array< Double, InputLayerType::inputSize >;
    using Output = std::array< Double, OutputLayerType::outputSize >;

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
        static_assert( A::outputSize == B::inputSize, "Cell sizes must match" );

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

    void randomizeWeights( Double min, Double max ) {
        _randomizeWeights( min, max, std::integral_constant< int, _layerCount >() );
    }

    void _randomizeWeights( Double min, Double max, std::integral_constant< int, 0 >)
    {}

    template < class I >
    void _randomizeWeights( Double min, Double max, I ) {
        std::get< I::value - 1 >( _layers ).randomizeWeights( min, max );
        _randomizeWeights( min, max, std::integral_constant< int, I::value - 1 >() );
    }

    template < class Out >
    void _backPropagate( Layers& layers, Layers* prev,
        std::tuple< GetContext< Cells >... >& ctx, Out& expected,
        std::integral_constant< int, 1 > )
    {
        auto& l = std::get< 0 >( layers );
        auto* p = prev ? &std::get< 0 >( *prev ) : nullptr;
        auto& c = std::get< 0 >( ctx );
        l.backPropagate( expected.data(), p ? p->_memory.data() : nullptr, c );
    }

    template < class I, class Out >
    auto _backPropagate( Layers& layers, Layers* prev,
        std::tuple< GetContext< Cells >... >& ctx, Out& expected, I )
    {
        auto& l = std::get< I::value - 1 >( layers );
        auto* p = prev ? &std::get< I::value - 1 >( *prev ) : nullptr;
        auto& c = std::get< I::value - 1 >( ctx );
        auto in = l.backPropagate( expected.data(),
            p ? p->_memory.data() : nullptr,  c );
        biElementWise( l._input.begin(), l._input.end(), in.begin(), in.begin(),
            std::plus< Double >() );
        _backPropagate( layers, prev, ctx, in, std::integral_constant< int, I::value - 1 >() );
        return in.begin() + l._input.size();
    }

    void _updateWeights( Double step, std::tuple< GetContext< Cells >... >& ctx,
        std::integral_constant< int, 0 > )
    { }

    template < class I >
    void _updateWeights( Double step, std::tuple< GetContext< Cells >... >& ctx, I ) {
        auto& dw = std::get< I::value - 1 >( ctx );
        auto& l = std::get< I::value - 1 >( _layers );
        l.adjustWeights( dw, step );
        _updateWeights( step, ctx, std::integral_constant< int, I::value - 1 >() );
    }

    void learn( const std::vector< Input >& sample, Output& out, Double step ) {
        std::vector< Layers > timeSteps;
        Layers *prev = &_layers;
        // Unroll forward propagation in time
        for ( const auto& frame : sample ) {
            timeSteps.push_back( *prev );
            prev = &timeSteps.back();
            auto& initial = std::get< 0 >( *prev );
            std::copy( frame.begin(), frame.end(), initial._input.begin() );
            forwardPropagate( *prev );
        }
        // Back propage the desired output
        std::tuple< GetContext< Cells >... > context;
        Output dH;
        biElementWise( out.begin(), out.end(),
            std::get< _layerCount - 1 >( *prev )._output.begin(), dH.begin(),
            [&]( Double rOut, Double eOut ) { return 2 * ( rOut - eOut ); } );
        for ( int i = timeSteps.size() - 1; i >= 0; i-- ) {
            auto r =_backPropagate( timeSteps[ i ], i ? &( timeSteps[ i - 1 ] ) : nullptr,
                context, dH, std::integral_constant< int, _layerCount >() );
            for ( int i = 0; i != dH.size(); i++ )
                dH[ i ] *= step;
        }
        // Propagate weight change
        _updateWeights( step, context, std::integral_constant< int, _layerCount >() );
    }


};

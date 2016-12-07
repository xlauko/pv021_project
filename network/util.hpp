#pragma once

#include <tuple>
#include <utility>

template< typename T, typename F, std::size_t... I >
void for_each( bool rev, T&& t, F f, std::index_sequence< I... > ) {
    if ( rev )
        ( f( std::get< sizeof...( I ) - I - 1 >( t ) ),... );
    else
        ( f( std::get< I >( t ) ),... );
}

template < typename... Ts, typename F >
void for_each_in_tuple( const std::tuple< Ts... >& t, F f, bool rev ) {
    for_each( rev, t, f, std::make_index_sequence< sizeof...(Ts) >() );
}

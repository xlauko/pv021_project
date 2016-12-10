#pragma once

#include <cassert>

// Minimal sufficient aray view implementation
template < class T, int Size >
struct ArrayView {
    ArrayView( T *array = nullptr ) : _array( array ) {}

    operator bool() {
        return _array;
    }

    const T *begin() const {
        assert( _array );
        return _array;
    }

    T *begin() {
        assert( _array );
        return _array;
    }

    const T *end() const {
        assert( _array );
        return _array + Size;
    }

    T *end() {
        assert( _array );
        return _array + Size;
    }

    T& operator[]( int i ) {
        assert( _array );
        assert( i < Size );
        return _array[ i ];
    }

    const T& operator[]( int i ) const {
        assert( i );
        assert( i < Size );
        return _array[ i ];
    }

    int size() {
        return Size;
    }

    T *_array;
};

template < class T, int size >
std::ostream& operator<<( std::ostream& o, const ArrayView< T, size >& l ) {
    bool first = true;
    for ( const T w : l ) {
        o << (first ? "" : ", ") << w;
        first = false;
    }
    return o;
}

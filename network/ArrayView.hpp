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

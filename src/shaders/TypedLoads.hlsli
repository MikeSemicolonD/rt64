//
// RT64
//

#pragma once

// WebGPU has no typed buffers: each Buffer<T> / RWBuffer<T> becomes a byte-address buffer holding the elements of the view RT64 binds, `width` bytes each.

uint loadUint(ByteAddressBuffer b, uint index, uint width) {
    const uint offset = index * width;
    const uint word = b.Load(offset & ~3u);
    if (width >= 4) {
        return word;
    }

    return (word >> ((offset & 3u) * 8u)) & ((1u << (width * 8u)) - 1u);
}

uint loadU16(ByteAddressBuffer b, uint index) {
    return loadUint(b, index, 2);
}

int loadS8(ByteAddressBuffer b, uint index) {
    return int(loadUint(b, index, 1) << 24) >> 24;
}

float loadF32(ByteAddressBuffer b, uint index) {
    return asfloat(b.Load(index * 4));
}

void storeUint(RWByteAddressBuffer b, uint index, uint width, uint value) {
    const uint offset = index * width;
    if (width >= 4) {
        b.Store(offset, value);
        return;
    }

    // Neighbouring threads share the word, so only this element's bits are cleared and set.
    const uint shift = (offset & 3u) * 8u;
    const uint mask = ((1u << (width * 8u)) - 1u) << shift;
    b.InterlockedAnd(offset & ~3u, ~mask);
    b.InterlockedOr(offset & ~3u, (value << shift) & mask);
}

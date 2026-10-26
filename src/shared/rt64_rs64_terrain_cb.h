//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Push constants of the RS64 whole-map terrain pass (RS64TerrainVS/PS). The map is a lattice of 4 samples per cell edge; record space is
    // (cell - origin) * 512 like the game's terrain records, then the anchor call's world and view-projection transforms and viewport.
    struct RS64TerrainCB {
        uint mapWidth;
        uint mapHeight;
        uint worldIndex;
        uint viewProjIndex;
        int originX;
        int originZ;
        uint fogIndex;
        uint sub;
        float2 screenScale;
        float2 screenOffset;
        float2 resolution;
        float heightScale;
        uint chunkCells;
        float4 fogColor;
        float offsetX;
        float offsetZ;
        uint chunksX;
        uint debug;
        uint smooth;
        // This draw's LOD (quad size (1 << lod) / sub lattice steps) and where its chunk list starts in gChunks.
        uint lod;
        uint chunkListBase;
        uint chunkCount;
        // 1: the water draw, one flat quad per water or shore cell at the water level (vertex id / 6 = cell), over the map plus waterMargin cells
        // each side, where the edge cell carries on outward and the outer ring reaches past any far plane.
        uint water;
        uint waterMargin;
    };
#ifdef HLSL_CPU
};
#endif

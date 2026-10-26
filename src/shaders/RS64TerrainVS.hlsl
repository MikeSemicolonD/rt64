//
// RT64
//

#include "RS64Terrain.hlsli"

// How far (cells) the water's outer ring reaches past the margin. Mirrors rs64lights::TerrainWaterFarCells.
static const uint kWaterFarCells = 4096;

float3 latticeColor(int2 p, uint latticeW, uint latticeH) {
    p = clamp(p, int2(0, 0), int2(latticeW - 1, latticeH - 1));
    return unpackRGBA8(gColors[p.y * latticeW + p.x]).rgb;
}

// The game's vertex colours on the same Catmull-Rom curve as the heights, so shading has no creases at the sample lines either.
float3 smoothColor(int2 i0, float2 t, uint latticeW, uint latticeH) {
    const float4 wx = catmullRomWeights(t.x), wy = catmullRomWeights(t.y);
    float3 c = 0.0f;
    for (int y = 0; y < 4; y++) {
        const float3 row = wx.x * latticeColor(i0 + int2(-1, y - 1), latticeW, latticeH) + wx.y * latticeColor(i0 + int2(0, y - 1), latticeW, latticeH)
                         + wx.z * latticeColor(i0 + int2(1, y - 1), latticeW, latticeH) + wx.w * latticeColor(i0 + int2(2, y - 1), latticeW, latticeH);
        c += wy[y] * row;
    }

    return saturate(c);
}

// Height and colour of the ground at lattice coordinate f.
float4 surfacePoint(float2 f, uint latticeW, uint latticeH) {
    const uint2 i0 = min(uint2(max(f, 0.0f)), uint2(latticeW - 2, latticeH - 2));
    const float2 t = f - float2(i0);
    const uint a = i0.y * latticeW + i0.x;
    float3 c;
    if (gTerrain.smooth) {
        c = smoothColor(int2(i0), t, latticeW, latticeH);
    }
    else {
        c = lerp(lerp(unpackRGBA8(gColors[a]), unpackRGBA8(gColors[a + 1]), t.x), lerp(unpackRGBA8(gColors[a + latticeW]), unpackRGBA8(gColors[a + latticeW + 1]), t.x), t.y).rgb;
    }

    return float4(surfaceHeight(f, latticeW, latticeH), c);
}

uint chunkLod(int2 c) {
    const uint chunksZ = gTerrain.chunkCount / gTerrain.chunksX;
    if ((c.x < 0) || (c.y < 0) || (uint(c.x) >= gTerrain.chunksX) || (uint(c.y) >= chunksZ)) {
        return 0xFF;
    }

    return gChunks[uint(c.y) * gTerrain.chunksX + uint(c.x)];
}

// The terrain draw: one instance per visible chunk of this draw's LOD (chunkCells * 4 * sub >> lod quads per edge), two triangles per quad (no
// index buffer), in fine units (sub per lattice step); an edge vertex next to a coarser chunk lies on that chunk's edge segment, so no T-junction opens.
// The water draw (gTerrain.water): one flat quad per water or shore cell at the water level; where the ground rises above it, the depth test cuts the shore.
// Positions follow RSPProcessCS (world then view-projection, N64 viewport) and RasterVS (screen to clip, the anchor draw's screen scale/offset).
void VSMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID,
    out float4 oPosition : SV_POSITION, out float4 oShade : COLOR0, out float2 oUV : TEXCOORD0, nointerpolation out uint oLayer : LAYER,
    out float3 oViewPos : TEXCOORD1, nointerpolation out float3 oNormal : TEXCOORD2, nointerpolation out float3 oSunDir : TEXCOORD3, out float2 oLattice : TEXCOORD4, out float2 oEdge : TEXCOORD5)
{
    const uint latticeW = gTerrain.mapWidth * 4 + 1;
    const uint latticeH = gTerrain.mapHeight * 4 + 1;
    const uint corner = vertexId % 6;
    static const uint2 kCorners[6] = { uint2(0, 0), uint2(1, 1), uint2(0, 1), uint2(0, 0), uint2(1, 0), uint2(1, 1) };
    oLayer = 0;
    oShade = 0;
    oUV = 0;
    oViewPos = 0;
    oNormal = 0;
    oSunDir = 0;
    oLattice = 0;
    oEdge = 0;
    uint2 cell;
    float2 cellOrigin;
    float2 f;
    float h;
    float3 c;
    bool outside = false;
    float edgeWater = 1.0f;
    if (gTerrain.water) {
        // Past the map edge every cell is drawn, so the map never ends in a void.
        const int margin = int(gTerrain.waterMargin);
        const uint spanW = gTerrain.mapWidth + 2 * gTerrain.waterMargin;
        const int2 signedCell = int2((vertexId / 6) % spanW, (vertexId / 6) / spanW) - margin;
        cell = uint2(clamp(signedCell, int2(0, 0), int2(gTerrain.mapWidth - 1, gTerrain.mapHeight - 1)));
        const uint kind = gCellLayers[cell.y * gTerrain.mapWidth + cell.x] >> 16;
        outside = any(signedCell != int2(cell));
        if ((kind == 0) && !outside) {
            oPosition = float4(2.0f, 2.0f, 2.0f, 1.0f);
            return;
        }

        // Flat surface: the ported terrain lighting with an upright normal. The outer ring's far corners reach past any far plane.
        const int2 k = int2(kCorners[corner]);
        const int2 last = int2(gTerrain.mapWidth, gTerrain.mapHeight) + margin - 1;
        const int2 reach = int2((signedCell.x == -margin) && (k.x == 0) ? -1 : ((signedCell.x == last.x) && (k.x == 1) ? 1 : 0),
                              (signedCell.y == -margin) && (k.y == 0) ? -1 : ((signedCell.y == last.y) && (k.y == 1) ? 1 : 0));
        cellOrigin = float2(signedCell);
        f = float2(signedCell * 4 + k * 4 + reach * int(kWaterFarCells * 4));
        h = gFrame[1].w * 16.0f - kWaterLift;
        const float ndl = gFrame[2].y;
        c = saturate((gFrame[0].rgb + ((ndl > 0.0f) ? gFrame[1].rgb * ndl : 0.0f)) / 255.0f);

        // Past the map edge the edge cell carries on outward: its texture and its ground where that lies above the water, with the water look as far as
        // the water draw covers that cell in the map (none for a ground cell, which an ocean level's sea often is: ground at the water level under waves).
        if (outside) {
            const float4 edge = surfacePoint(clamp(float2(signedCell * 4 + k * 4), 0.0f, float2(latticeW - 1, latticeH - 1)), latticeW, latticeH);
            edgeWater = (kind == 0) ? 0.0f : waterShoreAlpha(float2(signedCell * 4 + k * 4));
            h = (kind == 0) ? edge.x : min(h, edge.x);
            c = lerp(edge.yzw, c, edgeWater);
        }
    }
    else {
        const uint step = 1u << gTerrain.lod;
        const uint chunkFine = gTerrain.chunkCells * 4 * gTerrain.sub;
        const uint quadsPerChunk = chunkFine / step;
        const uint quad = vertexId / 6;
        const uint chunkId = gChunks[gTerrain.chunkListBase + instanceId];
        const uint2 chunk = uint2(chunkId % gTerrain.chunksX, chunkId / gTerrain.chunksX);
        const uint2 qi = uint2(quad % quadsPerChunk, quad / quadsPerChunk);
        const uint2 chunkBase = chunk * chunkFine;
        const uint2 F = chunkBase + (qi + kCorners[corner]) * step;
        cell = (chunkBase + qi * step) / (4 * gTerrain.sub);
        cellOrigin = float2(cell);
        if ((cell.x >= gTerrain.mapWidth) || (cell.y >= gTerrain.mapHeight)) {
            oPosition = float4(2.0f, 2.0f, 2.0f, 1.0f);
            return;
        }

        const float sub = float(gTerrain.sub);
        f = float2(F) / sub;
        float4 hc = surfacePoint(f, latticeW, latticeH);
        const uint2 local = F - chunkBase;
        if ((local.x == 0) || (local.x == chunkFine)) {
            const uint nl = chunkLod(int2(chunk) + int2((local.x == 0) ? -1 : 1, 0));
            const uint coarse = 1u << nl;
            if ((nl != 0xFF) && (nl > gTerrain.lod) && ((F.y % coarse) != 0)) {
                const uint z0 = (F.y / coarse) * coarse;
                hc = lerp(surfacePoint(float2(F.x, z0) / sub, latticeW, latticeH), surfacePoint(float2(F.x, z0 + coarse) / sub, latticeW, latticeH), float(F.y - z0) / float(coarse));
            }
        }

        if ((local.y == 0) || (local.y == chunkFine)) {
            const uint nl = chunkLod(int2(chunk) + int2(0, (local.y == 0) ? -1 : 1));
            const uint coarse = 1u << nl;
            if ((nl != 0xFF) && (nl > gTerrain.lod) && ((F.x % coarse) != 0)) {
                const uint x0 = (F.x / coarse) * coarse;
                hc = lerp(surfacePoint(float2(x0, F.y) / sub, latticeW, latticeH), surfacePoint(float2(x0 + coarse, F.y) / sub, latticeW, latticeH), float(F.x - x0) / float(coarse));
            }
        }

        h = hc.x;
        c = hc.yzw;
    }

    const uint cellWord = gCellLayers[cell.y * gTerrain.mapWidth + cell.x];

    const float3 pos = float3((f.x * 0.25f - gTerrain.originX) * 512.0f + gTerrain.offsetX, h * gTerrain.heightScale, (f.y * 0.25f - gTerrain.originZ) * 512.0f + gTerrain.offsetZ);
    const float4x4 world = gWorldTransforms[gTerrain.worldIndex];
    float4 tfPos = mul(mul(gViewProjTransforms[gTerrain.viewProjIndex], world), float4(pos, 1.0f));

    // The world transform is the game's modelview, so this is camera space. The heightmap's +y points down (water fills the largest heights)
    // and the game lights with that +y normal against the sun's travel direction, so both are negated: the open-air normal and the way to the sun.
    if (gTerrain.water) {
        oViewPos = mul(world, float4(pos, 1.0f)).xyz;
        // oEdge: x = past the map edge (no shoreline fade, fog by distance past the edge), y = how much of the water look its edge cell has (RS64TerrainPS).
        oLattice = f;
        oEdge = float2(outside ? 1.0f : 0.0f, edgeWater);
        oNormal = mul((float3x3)world, float3(0.0f, -1.0f, 0.0f));
        oSunDir = mul((float3x3)world, -gFrame[2].xyz);
    }

    const float fogAlpha = terrainFog(mul(world, float4(pos, 1.0f)).xyz);
    if (tfPos.w == 0.0f) {
        tfPos.w = 1e-6f;
    }

    const RSPViewport viewport = gViewports[gTerrain.viewProjIndex];
    const float3 ndc = tfPos.xyz / float3(tfPos.w, -tfPos.w, tfPos.w);
    float4 p = float4(ndc * viewport.scale + viewport.translate, tfPos.w);
    p.xy -= gTerrain.resolution / 2.0f;
    p.xy /= float2(gTerrain.resolution.x / 2.0f, gTerrain.resolution.y / -2.0f);
    p.xyz *= p.w;
    p.xy = (p.xy * gTerrain.screenScale) + gTerrain.screenOffset * p.w;

    // Texture rows run opposite to the cell's z, as the game maps its terrain tiles.
    const float2 inCell = f * 0.25f - cellOrigin;
    oPosition = p;
    oShade = float4(c, fogAlpha);
    oUV = float2(inCell.x, 1.0f - inCell.y);
    const uint layer = cellWord & 0xFFFF;
    oLayer = uint(gFrame[3 + layer / 4][layer % 4]);
}

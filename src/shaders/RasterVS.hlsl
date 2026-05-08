//
// RT64
//

#include "shared/rt64_other_mode.h"
#include "shared/rt64_raster_params.h"
#include "shared/rt64_render_flags.h"

#include "FbRendererCommon.hlsli"
#include "Library.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

LIBRARY_EXPORT void RasterVS(const RenderParams rp, in float4 iPosition, in float2 iUV, in float4 iColor, out float4 oPosition, out float2 oUV, out float4 oSmoothColor, out float4 oFlatColor) {
    float4 ndcPos = iPosition;
    
    // Skip any sort of transformation on the coordinates when rendering rects.
    if (!renderFlagRect(rp.flags)) {
        ndcPos.xy -= float2(FbParams.resolution / 2);
        ndcPos.xy /= float2(FbParams.resolution.x / 2.0f, FbParams.resolution.y / -2.0f);
        ndcPos.xyz *= ndcPos.w;
    }
    
    // Add half-pixel offset for rasterization.
    ndcPos.xy += gConstants.halfPixelOffset * ndcPos.w;
    
    // Output a fixed depth value for the entire triangle.
    const OtherMode otherMode = { rp.omL, rp.omH };
    const bool copyMode = (otherMode.cycleType() == G_CYC_COPY);
    const bool zSourcePrim = (otherMode.zSource() == G_ZS_PRIM);
    if (!copyMode && zSourcePrim) {
        const uint instanceIndex = instanceRenderIndices[gConstants.renderIndex].instanceIndex;
        ndcPos.z = instanceRDPParams[instanceIndex].primDepth.x * ndcPos.w;
    }

    // ROGUESQ rescue: cinematic-mux NON-rect Triangle draws had vertex output
    // that the RS rejected (whole-screen forced-position test confirmed this:
    // overriding the position made every cinematic-mux triangle paint via PS).
    // Keep the original screen x/y so primitives land where the game intended,
    // but rescue z and w so RS doesn't depth-clip or NaN-discard:
    //   * if w is ~0, replace with 1.0 to avoid NaN after perspective divide
    //   * clamp z into [0, w] so depth-clip lets the triangle through
    // The intent is: explosion content appears at its true screen position
    // instead of being rejected. PS magenta override (RasterPS.hlsl) will
    // paint those fragments magenta so we can verify the screen-space layout.
    // To disable: change #if 1 to #if 0 and rebuild.
#if 1
    if (!renderFlagRect(rp.flags) &&
        (rp.ccL == 0xFC11FE23u || rp.ccL == 0xFC127FFFu ||
         rp.ccL == 0xFC11E623u || rp.ccL == 0xFC119623u ||
         rp.ccL == 0x7C6E58D9u || rp.ccL == 0xBCA318B9u ||
         rp.ccL == 0xFC580253u || rp.ccL == 0xFCCDFCCDu))
    {
        if (abs(ndcPos.w) < 0.0001f) {
            ndcPos.w = 1.0f;
            ndcPos.z = 0.5f;
        }
        // Clamp z to [0, w] so we satisfy D3D's depth-clip rule.
        if (ndcPos.w > 0.0f) {
            ndcPos.z = clamp(ndcPos.z, 0.0f, ndcPos.w);
        } else {
            // Negative w — flip so it's positive then clamp.
            ndcPos.w = abs(ndcPos.w);
            ndcPos.z = clamp(abs(ndcPos.z), 0.0f, ndcPos.w);
        }
    }
#endif

    oPosition = ndcPos;
    oUV = iUV;
    oSmoothColor = iColor;
    oFlatColor = iColor;
}

#if defined(DYNAMIC_RENDER_PARAMS)
RenderParams getRenderParams() {
    uint instanceIndex = instanceRenderIndices[gConstants.renderIndex].instanceIndex;
    return DynamicRenderParams[instanceIndex];
}
#elif defined(SPEC_CONSTANT_RENDER_PARAMS)
#   include "RenderParamsSpecConstants.hlsli"
#endif

#if defined(DYNAMIC_RENDER_PARAMS) || defined(SPEC_CONSTANT_RENDER_PARAMS)
void VSMain(
    in float4 iPosition : POSITION
    , in float2 iUV : TEXCOORD
    , in float4 iColor : COLOR
    , out float4 oPosition : SV_POSITION
    , out float2 oUV : TEXCOORD
    , out float4 oSmoothColor : COLOR0
#if defined(DYNAMIC_RENDER_PARAMS) || defined(VERTEX_FLAT_COLOR)
    , out float4 oFlatColor : COLOR1
#endif
)
{
#if !defined(DYNAMIC_RENDER_PARAMS) && !defined(VERTEX_FLAT_COLOR)
    float4 oFlatColor;
#endif
    RasterVS(getRenderParams(), iPosition, iUV, iColor, oPosition, oUV, oSmoothColor, oFlatColor);
}
#endif
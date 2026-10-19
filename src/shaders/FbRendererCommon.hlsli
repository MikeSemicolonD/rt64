//
// RT64
//

#pragma once

#include "shared/rt64_fb_common.h"
#include "shared/rt64_frame_params.h"
#include "shared/rt64_framebuffer_params.h"
#include "shared/rt64_gpu_tile.h"
#include "shared/rt64_rdp_params.h"
#include "shared/rt64_rdp_tile.h"
#include "shared/rt64_render_indices.h"
#include "shared/rt64_render_params.h"

// Set 0 - Global resources and Enhancements (Raytracing or other).
ConstantBuffer<FrameParams> FrParams : register(b1, space0);
StructuredBuffer<RDPParams> instanceRDPParams : register(t2, space0);
StructuredBuffer<RDPTile> RDPTiles : register(t3, space0);
StructuredBuffer<GPUTile> GPUTiles : register(t4, space0);
StructuredBuffer<RenderIndices> instanceRenderIndices : register(t5, space0);
StructuredBuffer<RenderParams> DynamicRenderParams : register(t6, space0);
SamplerState gLinearWrapWrapSampler : register(s7, space0);
SamplerState gLinearWrapMirrorSampler : register(s8, space0);
SamplerState gLinearWrapClampSampler : register(s9, space0);
SamplerState gLinearMirrorWrapSampler : register(s10, space0);
SamplerState gLinearMirrorMirrorSampler : register(s11, space0);
SamplerState gLinearMirrorClampSampler : register(s12, space0);
SamplerState gLinearClampWrapSampler : register(s13, space0);
SamplerState gLinearClampMirrorSampler : register(s14, space0);
SamplerState gLinearClampClampSampler : register(s15, space0);
SamplerState gNearestWrapWrapSampler : register(s16, space0);
SamplerState gNearestWrapMirrorSampler : register(s17, space0);
SamplerState gNearestWrapClampSampler : register(s18, space0);
SamplerState gNearestMirrorWrapSampler : register(s19, space0);
SamplerState gNearestMirrorMirrorSampler : register(s20, space0);
SamplerState gNearestMirrorClampSampler : register(s21, space0);
SamplerState gNearestClampWrapSampler : register(s22, space0);
SamplerState gNearestClampMirrorSampler : register(s23, space0);
SamplerState gNearestClampClampSampler : register(s24, space0);

#ifdef RT64_TEXTURE_SLOTS
// Sets 1 and 2 - the draw's tiles 0-7 (SM 5.0 can't index resource arrays at run time; see TextureSampler.hlsli).
// Vulkan builds use 4 TMEM slots so 8 + 4 + 3 other textures fit a 16-texture stage (Raspberry Pi V3D).
#ifndef RT64_TMEM_SLOTS
#define RT64_TMEM_SLOTS 8
#endif
Texture2D<float4> gTextures[8] : register(t0, space1);
Texture1D<uint> gTMEM[RT64_TMEM_SLOTS] : register(t0, space2);

// RasterPS sets the tile slot before sampling; each switch folds to one case once the slot is a constant.
static uint gTextureSlot = 0;
#define RT64_SLOT_SWITCH(CASE) switch (gTextureSlot) { case 0: CASE(0) case 1: CASE(1) case 2: CASE(2) case 3: CASE(3) case 4: CASE(4) case 5: CASE(5) case 6: CASE(6) default: CASE(7) }
#if RT64_TMEM_SLOTS == 4
#define RT64_TMEM_SLOT_SWITCH(CASE) switch (gTextureSlot) { case 0: CASE(0) case 1: CASE(1) case 2: CASE(2) default: CASE(3) }
#else
#define RT64_TMEM_SLOT_SWITCH RT64_SLOT_SWITCH
#endif
#else
// Set 1 - RGBA32 texture cache.
Texture2D<float4> gTextures[8192] : register(t0, space1);

// Set 2 - TMEM texture cache.
Texture1D<uint> gTMEM[8192] : register(t0, space2);
#endif

// Set 3 - Framebuffer.
ConstantBuffer<FramebufferParams> FbParams : register(b0, space3);

//
// RT64
//

// FXAA following Timothy Lottes' FXAA 3.11 quality path: find the local edge from luma contrast, search along it, then blend across it.

struct FxaaCB {
    float2 invResolution;
    float2 padding;
};

[[vk::push_constant]] ConstantBuffer<FxaaCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
SamplerState gSampler : register(s2);

static const float EdgeThreshold = 0.166f;
static const float EdgeThresholdMin = 0.0833f;

// Lower than FXAA's usual 0.75 so N64 textures stay sharp.
static const float SubpixelQuality = 0.5f;

static const uint SearchSteps = 10;
static const float SearchStepSizes[SearchSteps] = { 1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 8.0f };

float Luma(float3 color) {
    return dot(color, float3(0.299f, 0.587f, 0.114f));
}

float LumaAt(float2 uv) {
    return Luma(gInput.SampleLevel(gSampler, uv, 0.0f).rgb);
}

float LumaOffset(float2 uv, float2 offset) {
    return LumaAt(uv + offset * gConstants.invResolution);
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const float2 px = gConstants.invResolution;
    const float3 rgbM = gInput.SampleLevel(gSampler, uv, 0.0f).rgb;
    const float lumaM = Luma(rgbM);
    float lumaN = LumaOffset(uv, float2(0.0f, -1.0f));
    float lumaS = LumaOffset(uv, float2(0.0f, 1.0f));
    const float lumaE = LumaOffset(uv, float2(1.0f, 0.0f));
    const float lumaW = LumaOffset(uv, float2(-1.0f, 0.0f));
    const float lumaMax = max(max(max(lumaN, lumaS), max(lumaE, lumaW)), lumaM);
    const float lumaMin = min(min(min(lumaN, lumaS), min(lumaE, lumaW)), lumaM);
    const float range = lumaMax - lumaMin;
    if (range < max(EdgeThresholdMin, lumaMax * EdgeThreshold)) {
        return float4(rgbM, 1.0f);
    }

    const float lumaNW = LumaOffset(uv, float2(-1.0f, -1.0f));
    const float lumaNE = LumaOffset(uv, float2(1.0f, -1.0f));
    const float lumaSW = LumaOffset(uv, float2(-1.0f, 1.0f));
    const float lumaSE = LumaOffset(uv, float2(1.0f, 1.0f));

    // Decide whether the edge runs horizontally or vertically.
    const float lumaNS = lumaN + lumaS;
    const float lumaWE = lumaW + lumaE;
    const float lumaNESE = lumaNE + lumaSE;
    const float lumaNWNE = lumaNW + lumaNE;
    const float lumaNWSW = lumaNW + lumaSW;
    const float lumaSWSE = lumaSW + lumaSE;
    const float edgeHorz = abs(-2.0f * lumaW + lumaNWSW) + abs(-2.0f * lumaM + lumaNS) * 2.0f + abs(-2.0f * lumaE + lumaNESE);
    const float edgeVert = abs(-2.0f * lumaS + lumaSWSE) + abs(-2.0f * lumaM + lumaWE) * 2.0f + abs(-2.0f * lumaN + lumaNWNE);
    const bool horzSpan = edgeHorz >= edgeVert;
    if (!horzSpan) {
        lumaN = lumaW;
        lumaS = lumaE;
    }

    // Pick the side of the edge with the larger gradient.
    float lengthSign = horzSpan ? px.y : px.x;
    const float gradientN = lumaN - lumaM;
    const float gradientS = lumaS - lumaM;
    const bool pairN = abs(gradientN) >= abs(gradientS);
    const float gradientScaled = max(abs(gradientN), abs(gradientS)) * 0.25f;
    const float lumaLocalAverage = 0.5f * (pairN ? (lumaN + lumaM) : (lumaS + lumaM));
    if (pairN) {
        lengthSign = -lengthSign;
    }

    // Walk along the edge in both directions until the luma leaves the edge's average.
    float2 posB = uv;
    float2 offNP;
    if (horzSpan) {
        posB.y += lengthSign * 0.5f;
        offNP = float2(px.x, 0.0f);
    }
    else {
        posB.x += lengthSign * 0.5f;
        offNP = float2(0.0f, px.y);
    }

    float2 posN = posB - offNP * SearchStepSizes[0];
    float2 posP = posB + offNP * SearchStepSizes[0];
    float lumaEndN = LumaAt(posN) - lumaLocalAverage;
    float lumaEndP = LumaAt(posP) - lumaLocalAverage;
    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    [unroll]
    for (uint i = 1; i < SearchSteps; i++) {
        if (!doneN) {
            posN -= offNP * SearchStepSizes[i];
            lumaEndN = LumaAt(posN) - lumaLocalAverage;
            doneN = abs(lumaEndN) >= gradientScaled;
        }

        if (!doneP) {
            posP += offNP * SearchStepSizes[i];
            lumaEndP = LumaAt(posP) - lumaLocalAverage;
            doneP = abs(lumaEndP) >= gradientScaled;
        }
    }

    // Offset toward the nearer end of the edge, only when the luma there crosses back over the average.
    const float dstN = horzSpan ? (uv.x - posN.x) : (uv.y - posN.y);
    const float dstP = horzSpan ? (posP.x - uv.x) : (posP.y - uv.y);
    const bool directionN = dstN < dstP;
    const bool lumaMLTZero = (lumaM - lumaLocalAverage) < 0.0f;
    const bool goodSpan = directionN ? ((lumaEndN < 0.0f) != lumaMLTZero) : ((lumaEndP < 0.0f) != lumaMLTZero);
    const float pixelOffset = goodSpan ? (0.5f - min(dstN, dstP) / (dstN + dstP)) : 0.0f;

    // Sub-pixel smoothing for thin features the edge search misses.
    const float subpixA = saturate(abs((2.0f * (lumaNS + lumaWE) + lumaNWSW + lumaNESE) / 12.0f - lumaM) / range);
    const float subpixB = (-2.0f * subpixA + 3.0f) * subpixA * subpixA;
    const float subpixOffset = subpixB * subpixB * SubpixelQuality;

    const float finalOffset = max(pixelOffset, subpixOffset);
    float2 finalUv = uv;
    if (horzSpan) {
        finalUv.y += finalOffset * lengthSign;
    }
    else {
        finalUv.x += finalOffset * lengthSign;
    }

    return float4(gInput.SampleLevel(gSampler, finalUv, 0.0f).rgb, 1.0f);
}

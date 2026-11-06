//
// RT64
//

#include "shared/rt64_rs64_shadows_cb.h"
#include "RS64CutoutAlpha.hlsli"
#include "RS64TerrainNormal.hlsli"

ConstantBuffer<RS64ShadowsCB> gShadows : register(b1, space3);

// Water reflection: tangent of its blur cone and its strength (the indirect target multiplies the water's colour).
static const float kWaterReflCone = 0.02f;
static const float kWaterReflStrength = 1.5f;
// Mirrors kWaterLift in RS64Terrain.hlsli: the drawn water sits this far (map units) over the water level and fades out as a shore rises to it.
static const float kWaterLift = 16.0f;
Texture2D<float> gDepth : register(t2, space3);

// The game's vertex fog at camera-space point p (RSPProcessCS): 0 clear, 1 fully fogged; 0 when the scene has no fog.
float rs64FogAlpha(float3 p) {
    if (gShadows.fogParams.z <= 0.0f) {
        return 0.0f;
    }
    const float4 c = mul(gShadows.screenFromView, float4(p, 1.0f));
    return (c.w > 0.0f) ? saturate(((max(c.z, 0.0f) / c.w) * gShadows.fogParams.x + gShadows.fogParams.y) / 255.0f) : 1.0f;
}

// The fog the terrain pass gives water at camera-space point p: the vertex fog, or past the map edge its fade toward the far plane (farPlaneFade in
// RS64Terrain.hlsli), whichever is more. Without the far fade, shadows and reflections stayed dark on water the frame shows as haze.
float rs64WaterFog(float3 p) {
    const float4x4 m = gShadows.screenFromView;
    const float denom = dot(m[2].xyz - m[3].xyz, p);
    const float sFar = (denom > 0.0f) ? (m[3].w - m[2].w) / denom : 0.0f;
    const float farFade = (sFar > 0.0f) ? smoothstep(0.5f, 1.0f, 1.0f / sFar) : 0.0f;
    return max(rs64FogAlpha(p), farFade);
}

// Mirrors rs64lights::frustumExitT: where the ray leaves this frame's view (|x|, |y| <= w, w > 0).
float rs64FrustumExitT(float3 o, float3 d) {
    const float4 c0 = mul(gShadows.screenFromView, float4(o, 1.0f));
    const float4 cd = mul(gShadows.screenFromView, float4(d, 0.0f));
    const float a[5] = { c0.w - c0.x, c0.w + c0.x, c0.w - c0.y, c0.w + c0.y, c0.w };
    const float b[5] = { cd.w - cd.x, cd.w + cd.x, cd.w - cd.y, cd.w + cd.y, cd.w };
    float t = 1.0e30f;
    [unroll]
    for (int i = 0; i < 5; i++) {
        if (a[i] < 0.0f) {
            return 0.0f;
        }
        if (b[i] < 0.0f) {
            t = min(t, -a[i] / b[i]);
        }
    }
    return t;
}

// Off-screen casters from past frames: only past the point where the ray leaves this frame's view (inside it, everything is drawn now),
// skipping an optional sphere around each slot's camera (craft never enter a slot). Segment j = slot j/2, part j%2.
bool rs64HistoryBlocked(RayDesc ray, inout float hitT) {
    if (gShadows.historyMask == 0) {
        return false;
    }
    const float tStart = max(ray.TMin, rs64FrustumExitT(ray.Origin, ray.Direction));
    if (tStart >= ray.TMax) {
        return false;
    }
    [loop]
    for (uint j = 0; j < 8; j++) {
        const uint slot = j >> 1;
        const uint bit = 0x08u << slot;
        if ((gShadows.historyMask & bit) == 0) {
            continue;
        }
        // Mirrors rs64lights::raySphereSpan.
        const float3 oc = ray.Origin - gShadows.historyCam[slot].xyz;
        const float b = dot(oc, ray.Direction);
        const float disc = b * b - (dot(oc, oc) - gShadows.historyCam[slot].w * gShadows.historyCam[slot].w);
        float t0 = tStart;
        float t1 = ray.TMax;
        if (disc >= 0.0f) {
            const float sq = sqrt(disc);
            t0 = ((j & 1) == 0) ? tStart : max(tStart, -b + sq);
            t1 = ((j & 1) == 0) ? min(ray.TMax, -b - sq) : ray.TMax;
        }
        else if ((j & 1) != 0) {
            continue;
        }
        if (t0 >= t1) {
            continue;
        }
        RayDesc r = ray;
        r.TMin = t0;
        r.TMax = t1;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(gScene, RAY_FLAG_NONE, bit, r);
        q.Proceed();
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hitT = q.CommittedRayT();
            return true;
        }
    }
    return false;
}

// A trace range: a fraction of the receiver's view distance, or world units when negative (the hangar, see fillRS64Shadows).
float rs64Range(float param, float dist) {
    return (param < 0.0f) ? -param : (param * dist);
}

struct RS64Receiver {
    float3 p;
    float3 n;
    float dist;
    bool sky;
    bool terrain;
    bool water;
    // How much the drawn water covers the pixel: 1 on the water surface, fading across a shore as the water's own alpha does.
    float waterCover;
    // How much terrain's own sun terminator applies: none under water, and fading out on cliffs.
    float terminatorWeight;
    // On static terrain, how far along a sun ray the terrain itself starts to block (rs64SunHitTerrainReceiver); 0 elsewhere.
    float terrainSkip;
};

// The terrain's self-shadow skip, in terrain cells.
static const float kTerrainSelfSkipCells = 1.0f;

// How far past side-on (-N.L) a model face turned away from the sun reaches full shadow.
static const float kModelTerminatorBand = 0.15f;

// A model face turned away from the sun is in its own shadow, as dark as a cast one (rs64lights::Config::modelTerminator). Only with the static
// terrain lookup on, which tells terrain apart: drawn terrain's facets would saw-tooth.
bool rs64ModelTerminator(RS64Receiver rc, bool deck) {
    const uint f = gShadows.featureMask;
    return ((f & 32u) != 0u) && ((f & 1u) != 0u) && (gShadows.softParams.w != 0.0f) && !rc.sky && !rc.terrain && !deck && (dot(rc.n, gShadows.sunDir.xyz) <= 0.0f);
}

bool rs64ReceiverSunHit(RS64Receiver rc, RayDesc ray, out float hitT, out uint id) {
    if (rc.terrainSkip > 0.0f) {
        return rs64SunHitTerrainReceiver(ray, gShadows.cutoutDrawCount, rc.terrainSkip, hitT, id);
    }

    return rs64SunHit(ray, gShadows.cutoutDrawCount, hitT, id);
}

// A heightfield samples a cliff with a few stretched rows of cells: its smooth normal swings column by column, and with the sun skimming the face the
// terminator drew long saw-tooth spikes down it. Only gentle slopes get it (terrainParams.zw, rs64lights::Config::terminatorFlatLo/Hi); cliffs keep the baked shading.

// View position from depth, facet normal from its derivatives (ddx/ddy stay in uniform control flow), smooth heightfield normal on static terrain.
RS64Receiver rs64Receiver(float2 px) {
    RS64Receiver r;
    const float depth = gDepth.Load(int3(px, 0));
    const float2 rel = (px - gShadows.viewport.xy) / gShadows.viewport.zw;
    const float4 v4 = mul(gShadows.viewFromScreen, float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, depth, 1.0f));
    r.p = v4.xyz / v4.w;
    r.n = normalize(cross(ddy(r.p), ddx(r.p)));
    if (dot(r.n, r.p) > 0.0f) {
        r.n = -r.n;
    }
    // Sky pin (0x7FBE/0x7FBF) and cleared depth.
    r.sky = (depth >= 0.9980f);
    r.dist = length(r.p);
    r.terrain = false;
    r.water = false;
    r.waterCover = 0.0f;
    r.terminatorWeight = 0.0f;
    r.terrainSkip = 0.0f;
    if (!r.sky) {
        float3 nT;
        r.terrain = (gShadows.softParams.w != 0.0f) && rs64TerrainNormal(r.p, r.dist, gShadows.terrainParams.x, nT);
        if (r.terrain) {
            // Water: the lookup found the water surface, or seabed under the drawn water, which covers it as much as its shore fade does there.
            // A yes/no here flipped whole shore cells between a sloped and a flat normal (a checkerboard along every shore).
            if (gShadows.waterPrim > 0) {
                r.waterCover = (gRs64TerrainPrim >= gShadows.waterPrim) ? 1.0f : rs64WaterCover(gRs64TerrainY, kWaterLift);
            }

            r.water = (r.waterCover > 0.0f);
            const float3 up = (dot(gRs64TerrainUp, nT) < 0.0f) ? -gRs64TerrainUp : gRs64TerrainUp;
            const float3 nS = normalize(lerp(nT, up, r.waterCover));
            r.n = (dot(nS, r.p) > 0.0f) ? -nS : nS;
            r.terminatorWeight = (1.0f - r.waterCover) * smoothstep(gShadows.terrainParams.z, gShadows.terrainParams.w, abs(dot(nT, up)));
            r.terrainSkip = kTerrainSelfSkipCells * gRs64TerrainCell;
        }
    }
    return r;
}

// The frame as drawn so far (before this pass composes), single-sampled: the trace passes skip MSAA targets.
Texture2D<float4> gColor : register(t12, space3);

// Pixel and depth where clip-space point c lands; false behind the camera or off screen.
bool rs64ScreenAt(float4 c, out int2 px, out float2 ndc, out float depth) {
    px = int2(0, 0);
    ndc = float2(0.0f, 0.0f);
    depth = 1.0f;
    if (c.w <= 0.0f) {
        return false;
    }

    ndc = c.xy / c.w;
    if (any(abs(ndc) > 1.0f)) {
        return false;
    }

    const float2 rel = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    px = clamp(int2(gShadows.viewport.xy + rel * gShadows.viewport.zw), int2(gShadows.viewport.xy), int2(gShadows.viewport.xy + gShadows.viewport.zw) - 1);
    depth = gDepth.Load(int3(px, 0));
    return true;
}

// A reflection hit at camera-space point q as the frame draws it (lit and fogged by the game), when it is on screen and nothing is drawn in front of it.
// Where the frame draws something farther there (sky or haze), the hit lies past the game's draw range or fog, so the reflection shows what is drawn.
bool rs64ScreenHit(float3 q, out float3 rgb) {
    rgb = float3(0.0f, 0.0f, 0.0f);
    int2 px;
    float2 ndc;
    float depth;
    if (!rs64ScreenAt(mul(gShadows.screenFromView, float4(q, 1.0f)), px, ndc, depth)) {
        return false;
    }

    const float4 v4 = mul(gShadows.viewFromScreen, float4(ndc, depth, 1.0f));
    const float seen = (depth >= 0.9980f) ? 1.0e30f : length(v4.xyz / v4.w);
    const float want = length(q);
    if (seen < want * 0.95f) {
        return false;
    }

    rgb = gColor.Load(int3(px, 0)).rgb;
    return true;
}

// The drawn sky in direction d (camera space), with the horizon haze the game fogs it with; off screen it takes the nearest screen edge. Where something
// is drawn in front of the sky there (a craft the ray passed), it takes the first sky pixel above. With no sky pixel on that edge (a camera looking down
// at the sea), the haze the game draws along that edge (the fog colour; the dome average without fog): giving up there cut the water's reflection
// off along a hard line.
static const uint kSkySearchSteps = 8;
bool rs64ScreenSky(float3 d, out float3 rgb) {
    rgb = (gShadows.fogParams.z > 0.0f) ? gShadows.fogColor.rgb : gShadows.skyHorizon.rgb;
    float4 c = mul(gShadows.screenFromView, float4(d, 0.0f));
    if (c.w <= 0.0f) {
        return true;
    }

    c.xy = clamp(c.xy / c.w, -1.0f, 1.0f);
    c.w = 1.0f;
    const float stepY = (1.0f - c.y) / float(kSkySearchSteps);
    [loop]
    for (uint k = 0; k <= kSkySearchSteps; k++) {
        int2 px;
        float2 ndc;
        float depth;
        if (rs64ScreenAt(float4(c.x, c.y + stepY * float(k), 0.0f, 1.0f), px, ndc, depth) && (depth >= 0.9980f)) {
            rgb = gColor.Load(int3(px, 0)).rgb;
            return true;
        }
    }

    return true;
}

// What water reflection ray r meets past a translucent hit at t (water point p), translucent draws aside: the next hit as the frame draws it or its
// hit colour, or the drawn sky.
bool rs64WaterBehind(RayDesc r, float t, float3 p, out float3 rgb) {
    rgb = float3(0.0f, 0.0f, 0.0f);
    RayDesc next = r;
    next.TMin = t * 1.001f + gShadows.tMin;
    float t2;
    uint id2;
    uint prim2;
    const bool hit = rs64ClosestHit(next, gShadows.cutoutDrawCount, t2, id2, prim2);
    if (hit) {
        const float4 hc2 = rs64HitColor(id2, prim2, gShadows.hitParams);
        const float3 hp2 = next.Origin + next.Direction * t2;
        float3 seen;
        rgb = rs64ScreenHit(hp2, seen) ? seen : lerp(hc2.rgb, gShadows.fogColor.rgb, max(rs64FogAlpha(hp2), rs64FogAlpha(p)));
        return true;
    }

    return (gShadows.skyZenith.w > 0.0f) && rs64ScreenSky(next.Direction, rgb);
}

// Floor-facing receivers (hangar deck); mirrors rs64lights::receiverSkipped / reflectionReceiver.
bool rs64FloorReceiver(float3 n) {
    return (gShadows.receiverSkip.w > 0.0f) && (dot(n, gShadows.receiverSkip.xyz) > gShadows.receiverSkip.w);
}

#ifdef RS64_SHADOW_SOFT

// Mask (shadow, blur radius px, view distance code = distance / 64 or 65000 for sky, AO) and indirect light (GI + reflection rgb, reflection luminance).
struct RS64TraceOut {
    float4 mask : SV_TARGET0;
    float4 indirect : SV_TARGET1;
};

RS64TraceOut PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) {
    const RS64Receiver rc = rs64Receiver(pos.xy);
    RS64TraceOut o;
    o.mask = float4(0.0f, 0.0f, rc.sky ? 65000.0f : (rc.dist / 64.0f), 0.0f);
    o.indirect = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (rc.sky) {
        return o;
    }
    const float3 p = rc.p;
    const float3 n = rc.n;
    const float dist = rc.dist;
    const uint features = gShadows.featureMask;
    const float3 L = gShadows.sunDir.xyz;
    const float ndl = dot(n, L);
    const bool deck = rs64FloorReceiver(n);
    // Fixed per-pixel rotation (rs64lights::interleavedNoise): a still view never shimmers.
    const float rot = frac(52.9829189f * frac(dot(pos.xy, float2(0.06711056f, 0.00583715f))));
    const float3 origin = p + n * (gShadows.normalBias * dist);
    // The hangar deck has decals and layers (the landing circle) just above it: AO, GI and reflection rays from it skip everything within 1% of the
    // view distance in height, whatever their angle.
    const float deckSkip = deck ? 0.01f * dist : 0.0f;

    // Any-hit rays, one loop so the shader keeps few RayQuery objects: N sun-disk rays (mirrors rs64lights::sunDiskDir), then AO hemisphere rays
    // (rs64lights::cosineHemisphereDir / aoWeight). Floor-facing receivers keep the game's painted shadow; faces turned away from the sun skip the sun rays.
    const bool sunRays = ((features & 1u) != 0u) && !deck && (ndl > 0.0f);
    const uint sunCount = sunRays ? max(gShadows.softRays, 1u) : 0u;
    const uint aoCount = ((features & 2u) != 0u) ? max(uint(gShadows.aoParams.x), 1u) : 0u;
    const float aoRange = max(rs64Range(gShadows.aoParams.y, dist), gShadows.tMin * 2.0f);
    const float3 seed = (abs(L.x) < 0.9f) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    const float3 st = normalize(cross(L, seed));
    const float3 sb = cross(L, st);
    uint blocked = 0;
    float hitSum = 0.0f;
    float ao = 0.0f;
    [loop]
    for (uint k = 0; k < sunCount + aoCount; k++) {
        const bool isSun = (k < sunCount);
        RayDesc r;
        r.Origin = origin;
        r.TMin = gShadows.tMin;
        if (isSun) {
            float3 dir = L;
            if (sunCount > 1) {
                const float rr = sqrt((float(k) + 0.5f) / float(sunCount)) * gShadows.softParams.x;
                const float th = 2.39996323f * float(k) + 6.28318531f * rot;
                dir = normalize(L + (cos(th) * rr) * st + (sin(th) * rr) * sb);
            }
            r.Direction = dir;
            r.TMax = gShadows.sunTMax;
        }
        else {
            r.Direction = rs64CosineDir(n, k - sunCount, aoCount, rot);
            r.TMin = max(gShadows.tMin, deckSkip / max(dot(r.Direction, n), 0.05f));
            r.TMax = max(aoRange, r.TMin * 2.0f);
        }
        float t;
        uint id;
        bool rayHit;
        if (isSun) {
            rayHit = rs64ReceiverSunHit(rc, r, t, id);
        }
        else {
            rayHit = rs64SunHit(r, gShadows.cutoutDrawCount, t, id);
        }

        if (rayHit) {
            if (isSun) {
                blocked++;
                hitSum += t;
            }
            else {
                ao += saturate(1.0f - t / aoRange);
            }
        }
    }
    // Off-screen history once, on the central sun ray: its blockers are far, so their penumbra is wide anyway; a hit blocks every ray that missed.
    if ((sunCount > 0) && (blocked < sunCount)) {
        RayDesc r;
        r.Origin = origin;
        r.Direction = L;
        r.TMin = gShadows.tMin;
        r.TMax = gShadows.sunTMax;
        float hT = r.TMax;
        if (rs64HistoryBlocked(r, hT)) {
            hitSum += hT * float(sunCount - blocked);
            blocked = sunCount;
        }
    }
    float shadow = 0.0f;
    float radius = 0.0f;
    // Water keeps cast shadows but not the terminator: a flat sheet under a low sun would darken all over.
    const bool terminator = rc.terrain && (rc.terminatorWeight > 0.0f) && (gShadows.softParams.w > 1.5f) && ((features & 1u) != 0u) && !deck;
    if (sunCount > 0) {
        const float meanHitT = (blocked > 0) ? (hitSum / float(blocked)) : 0.0f;
        radius = (meanHitT > 0.0f) ? clamp(meanHitT * gShadows.softParams.x * gShadows.softParams.y / dist, 0.0f, gShadows.softParams.z) : 0.0f;
        const float blockedFrac = float(blocked) / float(sunCount);
        // Grazing rays skim low-poly terrain and streak: fade the shadow in as the surface turns toward the sun; terrain gets a terminator instead.
        const float grazeFade = saturate((ndl - 0.05f) / 0.25f);
        shadow = grazeFade * blockedFrac;
        if (terminator) {
            shadow = lerp(shadow, 1.0f - (1.0f - blockedFrac) * saturate(ndl / 0.25f), rc.terminatorWeight);
        }
    }
    else if (rs64ModelTerminator(rc, deck)) {
        shadow = saturate(-ndl / kModelTerminatorBand);
    }
    else if (terminator && (ndl <= 0.0f)) {
        // Terrain turned away from the sun is in its own shadow.
        shadow = rc.terminatorWeight;
    }
    if (aoCount > 0) {
        ao /= float(aoCount);
        if (ao > 0.0f) {
            radius = max(radius, gShadows.aoParams.w);
        }
    }

    // Closest-hit rays, one loop: GI hemisphere rays, then the hangar deck's rough or the water's near-mirror reflection ray. Hit colours come from
    // the per-draw table.
    const bool waterRefl = ((features & 16u) != 0u) && rc.water;
    // Water carries its reflection in the indirect target instead of GI (RS64ShadowBlurPS adds it on top of the drawn water).
    const uint giCount = (((features & 4u) != 0u) && !waterRefl) ? max(uint(gShadows.giParams.x), 1u) : 0u;
    const uint reflCount = ((((features & 8u) != 0u) && deck) || waterRefl) ? 1u : 0u;
    const float3 v = p / dist;
    float3 gi = float3(0.0f, 0.0f, 0.0f);
    float3 refl = float3(0.0f, 0.0f, 0.0f);
    float3 waterR = float3(0.0f, 0.0f, 0.0f);
    float waterF = 0.0f;
    [loop]
    for (uint g = 0; g < giCount + reflCount; g++) {
        const bool isRefl = (g >= giCount);
        RayDesc r;
        r.Origin = origin;
        r.Direction = isRefl ? rs64RoughReflect(v, n, waterRefl ? kWaterReflCone : gShadows.reflParams.x, rot) : rs64CosineDir(n, g, giCount, frac(rot + 0.5f));
        const float rise = max(dot(r.Direction, n), 0.05f);
        r.TMin = max(gShadows.tMin, deckSkip / rise);
        r.TMax = isRefl ? gShadows.tMax : max(rs64Range(gShadows.giParams.y, dist), r.TMin * 2.0f);
        float t;
        uint id;
        uint prim;
        // Translucent draws (beams, foam) are seen only in water reflections.
        gRs64TranslucentRays = isRefl && waterRefl;
        const bool hitAny = rs64ClosestHit(r, gShadows.cutoutDrawCount, t, id, prim);
        gRs64TranslucentRays = false;
        const float4 glass = !hitAny ? float4(0.0f, 0.0f, 0.0f, 0.0f) : rs64TranslucentHit(id, prim, gRs64HitBary, gShadows.cutoutDrawCount);
        if (!hitAny) {
            // Water whose ray hits nothing reflects the sky as the frame draws it there (the dome is not traced), or the haze where none is drawn.
            float3 sky;
            if (isRefl && waterRefl && (gShadows.skyZenith.w > 0.0f) && rs64ScreenSky(r.Direction, sky)) {
                waterR = sky;
                waterF = saturate(rs64Schlick(saturate(dot(-v, n)), 0.02f) * kWaterReflStrength) * rc.waterCover * gShadows.skyZenith.w;
                if (gShadows.debugMode == 9) {
                    refl += sky;
                }
            }

            continue;
        }
        const float4 hc = rs64HitColor(id, prim, gShadows.hitParams);
        if (isRefl && (hc.a < -0.5f)) {
            // The deck's own floor draws (landing circle, plates) are not reflected.
            continue;
        }
        if (isRefl && waterRefl) {
            // Water: a near mirror, weighted by its own Fresnel (F0 0.02) at every distance; mixed in below. A hit on screen takes its drawn pixel, which
            // the game has lit and fogged; off screen or hidden, its hit colour fogged as the view fogs it there and at least as much as the water.
            const float3 hitP = r.Origin + r.Direction * t;
            waterF = saturate(rs64Schlick(saturate(dot(-v, n)), 0.02f) * kWaterReflStrength) * rc.waterCover;
            if (glass.a > 0.0f) {
                // A translucent draw is not in the frame yet (drawn after this pass): its texel over what the ray meets behind it, by the texel's alpha.
                const float3 glow = lerp(glass.rgb, gShadows.fogColor.rgb, max(rs64FogAlpha(hitP), rs64FogAlpha(p)));
                float3 behind;
                if (rs64WaterBehind(r, t, p, behind)) {
                    waterR = lerp(behind, glow, glass.a);
                }
                else {
                    waterR = glow;
                    waterF *= glass.a;
                }
            }
            else {
                float3 seen;
                waterR = rs64ScreenHit(hitP, seen) ? seen : lerp(hc.rgb, gShadows.fogColor.rgb, max(rs64FogAlpha(hitP), rs64FogAlpha(p)));
            }
            if (gShadows.debugMode == 9) {
                refl += waterR;
            }
        }
        else if (isRefl) {
            // Rough matte deck: ordinary hits fade out by their height above the deck, emissive ones (lights, glow cards) give the broad sheen.
            const float range = max(rs64Range(gShadows.reflParams.y, dist), gShadows.tMin * 2.0f);
            const float fade = (hc.a > 0.5f) ? 1.0f : pow(saturate(1.0f - (t * rise - deckSkip) / range), 2.0f);
            // Debug 9: the raw hit colour; 10: the hit instance (red drawn, green terrain, blue cutout, white emissive, grey history).
            const float3 idColor = (id == 0) ? float3(1, 0, 0) : (id == 1) ? float3(0, 1, 0) : (id == 2) ? float3(0, 0, 1) : (id == 7) ? float3(1, 1, 1) : float3(0.5f, 0.5f, 0.5f);
            refl += (gShadows.debugMode == 10) ? idColor : (gShadows.debugMode == 9) ? hc.rgb : hc.rgb * fade * rs64Schlick(saturate(dot(-v, n)), 0.04f) * gShadows.reflParams.z;
        }
        else if (hc.a > 0.5f) {
            gi += hc.rgb * gShadows.reflParams.w;
        }
        else {
            // The hit is lit when the sun reaches it; starting just short of the hit, a surface facing away from the sun blocks its own sun ray.
            RayDesc sr;
            sr.Origin = r.Origin + r.Direction * (t * 0.98f);
            sr.Direction = L;
            sr.TMin = gShadows.tMin;
            sr.TMax = gShadows.sunTMax;
            float sT;
            uint sId;
            if (!rs64SunHit(sr, gShadows.cutoutDrawCount, sT, sId)) {
                gi += hc.rgb * gShadows.sunColor.rgb;
            }
        }
    }
    if (giCount > 0) {
        gi *= gShadows.giParams.z / float(giCount);
    }
    // A fogged surface shows the fog's colour, so its shadow, AO and bounce light fade with the fog in front of it.
    const float fogHere = lerp(rs64FogAlpha(p), rs64WaterFog(p), rc.waterCover);
    const float clear = 1.0f - fogHere;
    shadow *= clear * lerp(1.0f, gShadows.skyHorizon.w, rc.waterCover);
    waterR = lerp(waterR, gShadows.fogColor.rgb, fogHere);
    ao *= clear;
    gi *= clear;

    // Water mixes dst * (1 - F) + R * F: the compose keeps 1 - F of the drawn water (alpha carries -F) and a second, additive draw adds R * F (rgb).
    // The colour target is UNORM, so a ratio against the water's colour clipped every reflection brighter than the water to the same value.
    const bool waterMix = (waterF > 0.0f) && (gShadows.debugMode != 9);
    if (waterMix) {
        radius = max(radius, gShadows.terrainParams.y);
    }

    const float3 indirect = waterMix ? (waterR * waterF) : (gi + refl);
    if (any(gi > 0.0f) || (any(refl != 0.0f) && !rc.water)) {
        radius = max(radius, gShadows.giParams.w);
    }
    o.mask = float4(shadow, radius, o.mask.b, ao);

    // Negative alpha marks a water mix and carries its -F; otherwise it is the reflection luminance (debug 8).
    o.indirect = float4(indirect, waterMix ? -max(waterF, 1.0e-4f) : dot(refl, float3(0.3f, 0.59f, 0.11f)));
    return o;
}

#else

// Hard path: one sun ray, darkens the colour target directly (debug modes 1-4).
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const RS64Receiver rc = rs64Receiver(pos.xy);
    const bool debug = (gShadows.debugMode != 0);
    if (rc.sky) {
        return debug ? float4(0.2f, 0.4f, 1.0f, 1.0f) : float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    // Floor-facing receivers keep the game's painted shadow (hangar). Debug: cyan.
    if (rs64FloorReceiver(rc.n)) {
        return debug ? float4(0.0f, 0.8f, 0.8f, 1.0f) : float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    const float3 L = gShadows.sunDir.xyz;
    const float ndl = dot(rc.n, L);
    // Water and cliffs keep cast shadows but not the terminator (RS64Receiver::terminatorWeight).
    const bool terminator = rc.terrain && (rc.terminatorWeight > 0.0f) && (gShadows.softParams.w > 1.5f);
    // Terrain turned away from the sun is in its own shadow (debug: dark blue). Other faces keep their baked shading; tracing them only adds acne.
    if (ndl <= 0.0f) {
        if (rs64ModelTerminator(rc, false)) {
            const float fm = 1.0f - gShadows.strength * saturate(-ndl / kModelTerminatorBand) * (1.0f - rs64FogAlpha(rc.p));
            return debug ? float4(0.35f, 0.1f, 0.1f, 1.0f) : float4(fm, fm, fm, 1.0f);
        }
        if (terminator) {
            // Faded by the fog in front of it like the lit branch, or fogged-out islands and shore lips stayed fully dark.
            const float fa = 1.0f - gShadows.strength * rc.terminatorWeight * (1.0f - rs64FogAlpha(rc.p));
            return debug ? float4(0.1f, 0.1f, 0.35f, 1.0f) : float4(fa, fa, fa, 1.0f);
        }
        return debug ? float4(0.5f, 0.5f, 0.5f, 1.0f) : float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    // Grazing rays skim low-poly terrain and streak; fade the shadow in as the surface turns toward the sun.
    const float grazeFade = saturate((ndl - 0.05f) / 0.25f);
    RayDesc ray;
    ray.Origin = rc.p + rc.n * (gShadows.normalBias * rc.dist);
    ray.Direction = L;
    ray.TMin = gShadows.tMin;
    ray.TMax = gShadows.sunTMax;
    float hitT;
    uint id;
    bool hit = rs64ReceiverSunHit(rc, ray, hitT, id);
    bool casterHit = hit && (id != 1);
    // Debug 4 shows history hits in magenta (instance id 3).
    if (!hit && rs64HistoryBlocked(ray, hitT)) {
        hit = true;
        casterHit = true;
        gRs64HitInstance = 3;
    }
    const float castShadow = hit ? grazeFade : 0.0f;
    const float shadow = terminator ? lerp(castShadow, 1.0f - (hit ? 0.0f : 1.0f) * saturate(ndl / 0.25f), rc.terminatorWeight) : castShadow;
    // Debug mode 2: hits closer than 2% of the view distance (surface self-hits) in red.
    if ((gShadows.debugMode == 2) && hit && (shadow > 0.0f) && (hitT < 0.02f * rc.dist)) {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    // Debug mode 4: shadow by caster (red drawn, yellow cutout, magenta history, green static terrain; terrain darker the nearer the hit).
    if ((gShadows.debugMode == 4) && hit && (shadow > 0.0f)) {
        if (!casterHit) {
            const float gg = saturate(hitT / (0.25f * rc.dist));
            return float4(0.0f, 0.25f + 0.75f * gg, 0.0f, 1.0f);
        }
        if (gRs64HitInstance == 3) {
            return float4(1.0f, 0.0f, 1.0f, 1.0f);
        }
        return (gRs64HitInstance == 2) ? float4(1.0f, 0.9f, 0.0f, 1.0f) : float4(0.8f, 0.0f, 0.0f, 1.0f);
    }
    // Debug mode 3: shadow cast by an alpha-tested cutout in yellow.
    if ((gShadows.debugMode == 3) && hit && (shadow > 0.0f) && (gRs64HitInstance == 2)) {
        return float4(1.0f, 0.9f, 0.0f, 1.0f);
    }
    if (debug) {
        const float m = 1.0f - shadow;
        return float4(m, m, m, 1.0f);
    }
    const float f = 1.0f - gShadows.strength * shadow * (1.0f - lerp(rs64FogAlpha(rc.p), rs64WaterFog(rc.p), rc.waterCover));
    return float4(f, f, f, 1.0f);
}

#endif

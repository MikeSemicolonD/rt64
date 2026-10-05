//
// RT64
//

#include "shared/rt64_rs64_shadows_cb.h"
#include "RS64CutoutAlpha.hlsli"
#include "RS64TerrainNormal.hlsli"

ConstantBuffer<RS64ShadowsCB> gShadows : register(b1, space3);
Texture2D<float> gDepth : register(t2, space3);

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

struct RS64Receiver {
    float3 p;
    float3 n;
    float dist;
    bool sky;
    bool terrain;
};

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
    if (!r.sky) {
        float3 nT;
        r.terrain = (gShadows.softParams.w != 0.0f) && rs64TerrainNormal(r.p, r.dist, gShadows.terrainParams.x, nT);
        if (r.terrain) {
            r.n = (dot(nT, r.p) > 0.0f) ? -nT : nT;
        }
    }
    return r;
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
    const float aoRange = max(gShadows.aoParams.y * dist, gShadows.tMin * 2.0f);
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
            r.TMax = gShadows.tMax;
        }
        else {
            r.Direction = rs64CosineDir(n, k - sunCount, aoCount, rot);
            r.TMin = max(gShadows.tMin, deckSkip / max(dot(r.Direction, n), 0.05f));
            r.TMax = max(aoRange, r.TMin * 2.0f);
        }
        float t;
        uint id;
        if (rs64SunHit(r, gShadows.cutoutDrawCount, t, id)) {
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
        r.TMax = gShadows.tMax;
        float hT = r.TMax;
        if (rs64HistoryBlocked(r, hT)) {
            hitSum += hT * float(sunCount - blocked);
            blocked = sunCount;
        }
    }
    float shadow = 0.0f;
    float radius = 0.0f;
    const bool terminator = rc.terrain && (gShadows.softParams.w > 1.5f) && ((features & 1u) != 0u) && !deck;
    if (sunCount > 0) {
        const float meanHitT = (blocked > 0) ? (hitSum / float(blocked)) : 0.0f;
        radius = (meanHitT > 0.0f) ? clamp(meanHitT * gShadows.softParams.x * gShadows.softParams.y / dist, 0.0f, gShadows.softParams.z) : 0.0f;
        const float blockedFrac = float(blocked) / float(sunCount);
        // Grazing rays skim low-poly terrain and streak: fade the shadow in as the surface turns toward the sun; terrain gets a terminator instead.
        const float grazeFade = saturate((ndl - 0.05f) / 0.25f);
        shadow = terminator ? (1.0f - (1.0f - blockedFrac) * saturate(ndl / 0.25f)) : grazeFade * blockedFrac;
    }
    else if (terminator && (ndl <= 0.0f)) {
        // Terrain turned away from the sun is in its own shadow.
        shadow = 1.0f;
    }
    if (aoCount > 0) {
        ao /= float(aoCount);
        if (ao > 0.0f) {
            radius = max(radius, gShadows.aoParams.w);
        }
    }

    // Closest-hit rays, one loop: GI hemisphere rays, then the hangar deck's rough reflection ray. Hit colours come from the per-draw table.
    const uint giCount = ((features & 4u) != 0u) ? max(uint(gShadows.giParams.x), 1u) : 0u;
    const uint reflCount = (((features & 8u) != 0u) && deck) ? 1u : 0u;
    const float3 v = p / dist;
    float3 gi = float3(0.0f, 0.0f, 0.0f);
    float3 refl = float3(0.0f, 0.0f, 0.0f);
    [loop]
    for (uint g = 0; g < giCount + reflCount; g++) {
        const bool isRefl = (g >= giCount);
        RayDesc r;
        r.Origin = origin;
        r.Direction = isRefl ? rs64RoughReflect(v, n, gShadows.reflParams.x, rot) : rs64CosineDir(n, g, giCount, frac(rot + 0.5f));
        const float rise = max(dot(r.Direction, n), 0.05f);
        r.TMin = max(gShadows.tMin, deckSkip / rise);
        r.TMax = isRefl ? gShadows.tMax : max(gShadows.giParams.y * dist, r.TMin * 2.0f);
        float t;
        uint id;
        uint prim;
        if (!rs64ClosestHit(r, gShadows.cutoutDrawCount, t, id, prim)) {
            continue;
        }
        const float4 hc = rs64HitColor(id, prim, gShadows.hitParams);
        if (isRefl && (hc.a < -0.5f)) {
            // The deck's own floor draws (landing circle, plates) are not reflected.
            continue;
        }
        if (isRefl) {
            // Rough matte deck: ordinary hits fade out by their height above the deck, emissive ones (lights, glow cards) give the broad sheen.
            const float range = max(gShadows.reflParams.y * dist, gShadows.tMin * 2.0f);
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
            sr.TMax = gShadows.tMax;
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
    const float3 indirect = gi + refl;
    if (any(indirect > 0.0f)) {
        radius = max(radius, gShadows.giParams.w);
    }
    o.mask = float4(shadow, radius, o.mask.b, ao);
    o.indirect = float4(indirect, dot(refl, float3(0.3f, 0.59f, 0.11f)));
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
    const bool terminator = rc.terrain && (gShadows.softParams.w > 1.5f);
    // Terrain turned away from the sun is in its own shadow (debug: dark blue). Other faces keep their baked shading; tracing them only adds acne.
    if (ndl <= 0.0f) {
        if (terminator) {
            const float fa = 1.0f - gShadows.strength;
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
    ray.TMax = gShadows.tMax;
    float hitT;
    uint id;
    bool hit = rs64SunHit(ray, gShadows.cutoutDrawCount, hitT, id);
    bool casterHit = hit && (id != 1);
    // Debug 4 shows history hits in magenta (instance id 3).
    if (!hit && rs64HistoryBlocked(ray, hitT)) {
        hit = true;
        casterHit = true;
        gRs64HitInstance = 3;
    }
    const float shadow = terminator ? (1.0f - (hit ? 0.0f : 1.0f) * saturate(ndl / 0.25f)) : (hit ? grazeFade : 0.0f);
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
    const float f = 1.0f - gShadows.strength * shadow;
    return float4(f, f, f, 1.0f);
}

#endif

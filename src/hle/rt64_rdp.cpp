//
// RT64
//

#include "rt64_rdp.h"

#include <cassert>
#include <cstdlib>
#include <unordered_set>

#include "../include/rt64_extended_gbi.h"

#include "common/rt64_math.h"
#include "gbi/rt64_f3d.h"

#include "rt64_color_converter.h"
#include "rt64_interpreter.h"
#include "rt64_state.h"

#ifndef NDEBUG
//#   define ASSERT_LOAD_METHODS
//#   define LOG_FILLRECT_METHODS
//#   define LOG_TEXRECT_METHODS
//#   define LOG_TILE_METHODS
//#   define LOG_COLOR_DEPTH_IMAGE_METHODS
//#   define LOG_TEXTURE_IMAGE_METHODS
//#   define LOG_LOAD_METHODS
#endif

namespace RT64 {
    // RDP
    static uint8_t getCommandLength(uint32_t commandId) {
        if (commandId == (G_TEXRECT & 0x3F) || commandId == (G_TEXRECTFLIP & 0x3F)) {
            return 2;
        }
        else if (commandId >= G_RDPTRI_BASE && commandId <= (uint32_t)RDPTriangle::MaxValue) {
            // Determine the triangle type.
            bool shaded = (commandId & (uint32_t)RDPTriangle::Shaded) != 0;
            bool textured = (commandId & (uint32_t)RDPTriangle::Textured) != 0;
            bool z_buffered = (commandId & (uint32_t)RDPTriangle::Depth) != 0;

            // Determine the length of this triangle command based on the flags it has set.
            uint8_t commandLength = triangleBaseWords;

            if (shaded) {
                commandLength += triangleShadeWords;
            }

            if (textured) {
                commandLength += triangleTexWords;
            }

            if (z_buffered) {
                commandLength += triangleDepthWords;
            }

            return commandLength;
        }
        else {
            return 1;
        }
    }

    RDP::RDP(State *state) : state(state) {
        memset(TMEM, 0, sizeof(TMEM));
        crashed = false;
        crashReason = CrashReason::None;

        // Set up the command lengths array.
        for (uint32_t commandId = 0; commandId < commandWordLengths.size(); commandId++) {
            commandWordLengths[commandId] = getCommandLength(commandId);
        }
    }

    void RDP::setGBI() {
        gbi = state->ext.interpreter->gbiManager.getGBIForRDP();
    }

    void RDP::reset() {
        colorCombinerStackSize = 1;
        envColorStackSize = 1;
        primColorStackSize = 1;
        primDepthStackSize = 1;
        blendColorStackSize = 1;
        fogColorStackSize = 1;
        fillColorStackSize = 1;
        scissorStackSize = 1;
        colorCombinerStack[0] = { 0, 0 };
        otherMode = { 0, 0 };
        envColorStack[0] = {0.0f, 0.0f, 0.0f, 0.0f};
        primColorStack[0] = { 0.0f, 0.0f, 0.0f, 0.0f };
        primLODStack[0] = { 0.0f, 0.0f };
        primDepthStack[0] = { 0.0f, 0.0f };
        fogColorStack[0] = { 0.0f, 0.0f, 0.0f, 0.0f };
        fillColorStack[0] = 0;
        blendColorStack[0] = { 0.0f, 0.0f, 0.0f, 0.0f };
        keyCenter = { 0.0f, 0.0f, 0.0f };
        keyScale = { 0.0f, 0.0f, 0.0f };
        convertK[0] = 0;
        convertK[1] = 0;
        convertK[2] = 0;
        convertK[3] = 0;
        convertK[4] = 0;
        convertK[5] = 0;
        scissorRectStack[0] = { 0, 0, 8192, 8192 };
        scissorModeStack[0] = 0;
        pendingCommandCurrentBytes = 0;
        pendingCommandRemainingBytes = 0;

        clearExtended();
    }

    void RDP::crash(CrashReason reason) {
        assert(reason != CrashReason::None);
        crashReason = reason;
        crashed = true;
        assert(false && "Crash screen not implemented.");
    }
    
    void RDP::checkFramebufferPair() {
        if (colorImage.changed || depthImage.changed) {
            state->flush();
            state->submitFramebufferPair(colorImage.changed ? FramebufferPair::FlushReason::ColorImageChanged : FramebufferPair::FlushReason::DepthImageChanged);

            const int workloadCursor = state->ext.workloadQueue->writeCursor;
            Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
            workload.addFramebufferPair(colorImage.address, colorImage.fmt, colorImage.siz, colorImage.width, depthImage.address);
            colorImage.changed = false;
            depthImage.changed = false;

            state->updateDrawStatusAttribute(DrawAttribute::FramebufferPair);
        }
    }
    
    void RDP::checkFramebufferOverlap(uint32_t tmemStart, uint32_t tmemWords, uint32_t tmemMask, uint32_t addressStart, uint32_t addressEnd, uint32_t tileWidth, uint32_t tileHeight, bool RGBA32, bool makeTileCopy) {
        auto &fbManager = state->framebufferManager;
        Framebuffer *fb = fbManager.findMostRecentContaining(addressStart, addressEnd);
        if (fb != nullptr) {
            const bool gpuCopiesEnabled = state->ext.emulatorConfig->framebuffer.copyWithGPU;
            if (!gpuCopiesEnabled) {
                return;
            }

            FramebufferTile fbTile;
            bool couldMakeTile = false;
            if (makeTileCopy) {
                // Find the best possible fitting GPU copy to store in this TMEM region.
                couldMakeTile = fbManager.makeFramebufferTile(fb, addressStart, addressEnd, tileWidth, tileHeight, fbTile, RGBA32);
            }

            // Always tags regions in TMEM, regardless of whether it's possible to make a copy or not.
            uint32_t fbEnd = fb->addressStart + fb->imageRowBytes(fb->width) * fb->maxHeight;
            bool syncRequired = (fb->addressStart < addressEnd) && (fbEnd > addressStart);
            fbManager.insertRegionsTMEM(fb->addressStart, tmemStart, std::min(tmemWords, uint32_t(RDP_TMEM_WORDS)), tmemMask, RGBA32, syncRequired, couldMakeTile ? &regionIterators : nullptr);

            if (couldMakeTile) {
                // Make a new tile copy resource.
                const uint32_t newTileWidth = fbTile.right - fbTile.left;
                const uint32_t newTileHeight = fbTile.bottom - fbTile.top;
                uint64_t newTileId = fbManager.findTileCopyId(newTileWidth, newTileHeight);

                // If valid, store the FB tile and the copy ID in the relevant regions.
                for (FramebufferManager::RegionIterator regionIt : regionIterators) {
                    regionIt->fbTile = fbTile;
                    regionIt->tileCopyId = newTileId;
                }
                
                // Queue the operation to make the tile copy.
                FramebufferOperation fbOp = fbManager.makeTileCopyTMEM(newTileId, fbTile);
                state->drawFbOperations.emplace_back(fbOp);
            }
        }
    }
    
    void RDP::checkImageOverlap(uint32_t addressStart, uint32_t addressEnd) {
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        const FixedRect colorRect = fbPair.drawColorRect;
        if (colorRect.isEmpty()) {
            return;
        }
        
        const uint32_t imageWidth = fbPair.colorImage.width;
        const uint32_t colorRowStart = colorRect.top(false);
        const uint32_t colorRowEnd = colorRect.bottom(true);
        const uint32_t colorBpr = imageWidth << fbPair.colorImage.siz >> 1;
        const uint32_t colorStart = fbPair.colorImage.address + colorRowStart * colorBpr;
        const uint32_t colorEnd = colorStart + (colorRowEnd - colorRowStart) * colorBpr;
        bool overlapDetected = false;
        if ((addressStart < colorEnd) && (addressEnd > colorStart)) {
            colorImage.changed = true;
            overlapDetected = true;
        }
        else if (fbPair.depthWrite) {
            const FixedRect depthRect = fbPair.drawDepthRect;
            if (depthRect.isEmpty()) {
                return;
            }

            const uint32_t depthRowStart = depthRect.top(false);
            const uint32_t depthRowEnd = depthRect.bottom(true);
            const uint32_t depthBpr = imageWidth << G_IM_SIZ_16b >> 1;
            const uint32_t depthStart = fbPair.depthImage.address + depthRowStart * depthBpr;
            const uint32_t depthEnd = depthStart + (depthRowEnd - depthRowStart) * depthBpr;
            if ((addressStart < depthEnd) && (addressEnd > depthStart)) {
                depthImage.changed = true;
                overlapDetected = true;
            }
        }

        if (overlapDetected && (colorImage.changed || depthImage.changed)) {
            state->flush();
            state->submitFramebufferPair(colorImage.changed ? FramebufferPair::FlushReason::SamplingFromColorImage : FramebufferPair::FlushReason::SamplingFromDepthImage);
        }
    }
    
    int32_t RDP::movedFromOrigin(int32_t x, uint16_t ori) {
        if (ori < G_EX_ORIGIN_NONE) {
            return x + ((ori * colorImage.width * 4) / G_EX_ORIGIN_RIGHT);
        }
        else {
            return x;
        }
    };

    constexpr uint32_t ExtendedMask = 0x80000000U;

    uint32_t RDP::maskAddress(uint32_t address) {
        if (state->extended.extendRDRAM && ((address & ExtendedMask) == ExtendedMask)) {
            return address - ExtendedMask;
        }
        else {
            return address & RDP_ADDRESS_MASK;
        }
    }

    void RDP::setColorImage(uint8_t fmt, uint8_t siz, uint16_t width, uint32_t address) {
        // Make sure the new color image is actually different. Some games will set the color image
        // multiple times despite setting the exact same parameters.
        const uint32_t newAddress = maskAddress(address);
        if ((colorImage.fmt != fmt) ||
            (colorImage.siz != siz) ||
            (colorImage.width != width) ||
            (colorImage.address != newAddress))
        {
            { static int n=0; ++n;
                bool isZero = (newAddress == 0);
                if (isZero || n<=30 || (n%50)==0) {
                    if(false) fprintf(stderr, "[trace] setColorImage #%d addr=0x%08X w=%u fmt=%u siz=%u%s\n",
                        n, newAddress, (unsigned)width, (unsigned)fmt, (unsigned)siz,
                        isZero ? " <ZERO>" : "");
                    fflush(stderr);
                }
            }
            colorImage.fmt = fmt;
            colorImage.siz = siz;
            colorImage.width = width;
            colorImage.address = newAddress;
            colorImage.changed = true;

#       ifdef LOG_COLOR_DEPTH_IMAGE_METHODS
            RT64_LOG_PRINTF("RDP::setColorImage(fmt %u, siz %u, width %u, address 0x%08X)", fmt, siz, width, address);
#       endif
        }
    }

    void RDP::setDepthImage(uint32_t address) {
        const uint32_t newAddress = maskAddress(address);
        // Log distinct depth buffer addresses under ROGUESQ_LOG_DPC. Tells us
        // which of the "fbs" we found are actually Z-buffers (would render as
        // gradient if VI ever sampled them).
        {
            static const bool log_z = []{
                const char *a = std::getenv("ROGUESQ_LOG_ALL");
                if (a && *a && *a != '0') return true;
                const char *e = std::getenv("ROGUESQ_LOG_DPC");
                return e && *e && *e != '0';
            }();
            if (log_z) {
                static std::unordered_set<uint32_t> seen;
                if (seen.insert(newAddress).second) {
                    fprintf(stderr, "[zimg] new depth-image address 0x%08X (count=%zu)\n",
                        newAddress, seen.size());
                    fflush(stderr);
                }
            }
        }
        if (depthImage.address != newAddress) {
            depthImage.address = newAddress;
            depthImage.changed = true;

#       ifdef LOG_COLOR_DEPTH_IMAGE_METHODS
            RT64_LOG_PRINTF("RDP::setDepthImage(address 0x%08X)", address);
#       endif
        }
    }

    void RDP::setTextureImage(uint8_t fmt, uint8_t siz, uint16_t width, uint32_t address) {
        const uint32_t newAddr = maskAddress(address);
        // ROGUESQ_LOG_TEXBYTES — dump first 32 bytes at the texture address so
        // we can verify whether the cinematic sprite assets are actually loaded.
        // If the bytes are all 0x00 / 0xFF, the asset never got DMAd in →
        // upstream loading bug. If they look like pixel data, rasterization is
        // failing downstream.
        {
            static const bool log_tex = []{
                const char *e = std::getenv("ROGUESQ_LOG_TEXBYTES");
                if (e && *e && *e != '0') return true;
                const char *a = std::getenv("ROGUESQ_LOG_ALL");
                return a && *a && *a != '0';
            }();
            if (log_tex && newAddr != 0) {
                static int n = 0;
                static std::unordered_set<uint32_t> seen;
                bool fresh = seen.insert(newAddr).second;
                if (fresh && (++n <= 100)) {
                    const uint8_t *p = &state->RDRAM[newAddr];
                    fprintf(stderr, "[tex-bytes] addr=0x%08X w=%u fmt=%u siz=%u :",
                        newAddr, (unsigned)width, (unsigned)fmt, (unsigned)siz);
                    for (int i = 0; i < 32; ++i) {
                        fprintf(stderr, " %02X", p[i ^ 3]);  // BE swizzle for word-aligned access
                    }
                    fprintf(stderr, "\n");
                    fflush(stderr);
                }
            }
        }
        { static int n=0; ++n;
          bool isZero = (newAddr == 0);
          if (isZero || n<=10 || (n%5000)==0) {
              if(false) fprintf(stderr, "[trace] RDP::setTextureImage #%d addr=0x%08X w=%u fmt=%u siz=%u%s\n",
                  n, newAddr, (unsigned)width, (unsigned)fmt, (unsigned)siz,
                  isZero ? " <ZERO>" : "");
              fflush(stderr);
          } }
        texture.fmt = fmt;
        texture.siz = siz;
        texture.width = width;
        texture.address = newAddr;
        state->updateDrawStatusAttribute(DrawAttribute::Texture);

#   ifdef LOG_TEXTURE_IMAGE_METHODS
        RT64_LOG_PRINTF("RDP::setTextureImage(fmt %u, siz %u, width %u, address 0x%08X)", fmt, siz, width, address);
#   endif
    }

    void RDP::setCombine(uint64_t combine) {
        // EXPERIMENT 2026-05-05: Substitute the SHADE combiner input.
        // Memory-module + X-Wing 3D content uses TEXEL × SHADE combiner mode
        // (mux 0xFFFFFFFFFC127E24); the recompiled Factor5 ucode produces
        // SHADE values that include zeros at top vertices, making the upper
        // model black ("top of model is gone"). User-confirmed: replacing
        // this mode with TEXEL × PRIM_COLOR makes the model render but tints
        // everything with PRIM (X-Wing → sunset orange). Real fix is to make
        // the combiner output PURE TEXEL (no shade multiply, no prim tint).
        //
        // ROGUESQ_SHADE_FIX modes:
        //   1 = swap to Mode A (TEXEL × PRIM) — A/B verification, tinted
        //   2 = rewrite to DECAL (output = TEXEL only) — clean fix
        constexpr uint64_t kModeB_TexelShade = 0xFFFFFFFFFC127E24ULL;
        constexpr uint64_t kModeA_TexelPrim  = 0xFF2FFFFFFC119623ULL;
        // DECAL combiner: cycle output = (A-B)*C+D = (0-0)*0 + TEXEL0 = TEXEL0
        // Standard libultra G_CC_DECALRGB mux value (verified against gbi.h).
        constexpr uint64_t kDecal             = 0xFFFCF278FCFFFFFFULL;
        static const int experiment_mode = []() {
            const char *e = std::getenv("ROGUESQ_SHADE_FIX");
            // legacy: ROGUESQ_SWAP_SHADE=1 == ROGUESQ_SHADE_FIX=1
            const char *legacy = std::getenv("ROGUESQ_SWAP_SHADE");
            if (legacy && *legacy && *legacy != '0') return 1;
            if (e && *e) return atoi(e);
            return 0;
        }();
        if (experiment_mode) {
            uint64_t old = combine;
            const uint32_t L = static_cast<uint32_t>(combine & 0xFFFFFFFFULL);
            // SHADE = combiner input value 4. C input is 5 bits; cycle 0
            // C is L bits 15-19, cycle 1 C is L bits 0-4. The C input is
            // the multiplier in (A-B)*C+D, so SHADE in C makes output dark
            // when shade is dark. Detect either cycle.
            const uint32_t c0 = (L >> 15) & 0x1F;
            const uint32_t c1 = (L >> 0) & 0x1F;
            const bool shade_in_c = (c0 == 4) || (c1 == 4);
            const bool literal_match = (combine == kModeB_TexelShade);
            if (experiment_mode == 1 && literal_match) {
                combine = kModeA_TexelPrim;
            } else if (experiment_mode == 2 && literal_match) {
                combine = kDecal;
            } else if (experiment_mode == 3 && shade_in_c) {
                // Generalized: any combiner with SHADE in the multiplier
                // slot gets rewritten to DECAL (output = pure TEXEL).
                combine = kDecal;
            }
            if (combine != old) {
                static int n = 0;
                static std::unordered_set<uint64_t> seen;
                bool fresh = seen.insert(old).second;
                if (++n <= 3 || fresh) {
                    fprintf(stderr, "[exp mode=%d] rewrote 0x%016llX -> 0x%016llX (c0=%u c1=%u) #%d\n",
                        experiment_mode,
                        (unsigned long long)old,
                        (unsigned long long)combine, c0, c1, n);
                    fflush(stderr);
                }
            }
        }
        // 2026-05-06: Cinematic-particle alpha fix. The TEXRECT particle
        // combiner mux (lower 32 = 0xFC11FE23, color = TEXEL × PRIM) has
        // alpha-D = 7 (ZERO) in its canonical 0xFFFFFFFF_FC11FE23 variant.
        // With the cinematic blender configured for alpha-gated output, an
        // alpha=0 combined value collapses the texrect into the existing
        // framebuffer pixel — invisible particles. Surgically rewrite the
        // alpha-D field (H bits 11-9 cycle 0, H bits 2-0 cycle 1) from 7
        // to 1 (TEXEL0_ALPHA) so the per-pixel TLUT alpha drives blending.
        // This preserves color combiner (TEXEL × PRIM) intact.
        // Set ROGUESQ_PARTICLE_FIX=0 to disable.
        //
        // ROGUESQ_PARTICLE_DEBUG=1 adds a high-visibility override —
        // forces alpha-D = 6 (ONE) so particles render as opaque
        // rectangular blocks (no soft edges) regardless of texture alpha.
        // Used to confirm whether the geometry reaches the framebuffer
        // at all when subtle alpha rendering looks wrong.
        static const bool particle_fix = []() {
            const char *e = std::getenv("ROGUESQ_PARTICLE_FIX");
            return !e || atoi(e) != 0;  // default on
        }();
        static const bool particle_debug = []() {
            const char *e = std::getenv("ROGUESQ_PARTICLE_DEBUG");
            return e && atoi(e) != 0;  // default off
        }();
        if (particle_fix && (combine & 0xFFFFFFFFULL) == 0xFC11FE23ULL) {
            uint64_t old = combine;
            uint64_t H = (combine >> 32ULL) & 0xFFFFFFFFULL;
            const uint64_t aD0 = (H >> 9) & 0x7;
            const uint64_t aD1 = (H >> 0) & 0x7;
            // Diagnostic mode: force ONE for both cycles, regardless of
            // current value. Otherwise, only rewrite when alpha-D is
            // currently ZERO (==7) — preserving variant 0xFFFFF3F9 which
            // already has aD0=1=TEXEL0_ALPHA.
            const uint64_t target = particle_debug ? 6ULL : 1ULL;  // ONE vs TEXEL0_ALPHA
            if (particle_debug || aD0 == 7) H = (H & ~(0x7ULL << 9)) | (target << 9);
            if (particle_debug || aD1 == 7) H = (H & ~(0x7ULL << 0)) | (target << 0);
            combine = (combine & 0xFFFFFFFFULL) | (H << 32);
            if (combine != old) {
                static int n = 0;
                if (++n <= 3) {
                    fprintf(stderr, "[particle-fix%s] rewrote 0x%016llX -> 0x%016llX #%d\n",
                        particle_debug ? "/DEBUG" : "",
                        (unsigned long long)old, (unsigned long long)combine, n);
                    fflush(stderr);
                }
            }
        }
        // 2026-05-06: TEXEL × PRIM TEXRECT alpha=SHADE fix.
        //
        // The actual rendering mux for cinematic-particle TEXRECTs is
        // 0xFF2FFFFF_FC119623 (color = TEXEL × PRIM, alpha = SHADE × PRIM_A).
        // RT64's TEXRECT rasterizer doesn't populate SHADE for the fragment
        // (TEXRECTs have no vertex-stream shade attribute), so alpha = 0
        // regardless of PRIM_ALPHA. Confirmed by per-pixel trace: drawTexRect
        // fires with this mux 10000+ times during cinematic, valid coords,
        // valid colorAddr, but no visible output even with mode-2 VI override
        // forcing the rendered fb to be the displayed one.
        //
        // Rewrite alpha-A cycle 0 from 4 (SHADE) to 6 (ONE) so the alpha
        // cycle becomes (1 - 0) × PRIM_ALPHA + 0 = PRIM_ALPHA. With
        // PRIM_ALPHA = 0xFF (game's default), particles render fully opaque.
        // We accept the loss of soft-alpha gradient to confirm whether the
        // TEXRECTs ever produce visible pixels.
        // Set ROGUESQ_TEXRECT_ALPHA_FIX=0 to disable.
        static const bool texrect_alpha_fix = []() {
            const char *e = std::getenv("ROGUESQ_TEXRECT_ALPHA_FIX");
            return !e || atoi(e) != 0;  // default on
        }();
        if (texrect_alpha_fix && (combine & 0xFFFFFFFFULL) == 0xFC119623ULL) {
            uint64_t old = combine;
            uint64_t L = combine & 0xFFFFFFFFULL;
            // Alpha-A cycle 0 lives at L bits 12-14. Rewrite from 4 (SHADE)
            // to 6 (ONE).
            L = (L & ~(0x7ULL << 12)) | (6ULL << 12);
            combine = ((combine >> 32ULL) << 32ULL) | L;
            if (combine != old) {
                static int n = 0;
                if (++n <= 3) {
                    fprintf(stderr, "[texrect-alpha-fix] rewrote 0x%016llX -> 0x%016llX #%d\n",
                        (unsigned long long)old, (unsigned long long)combine, n);
                    fflush(stderr);
                }
            }
        }
        // 2026-05-06: Cinematic pass-2 visibility debug.
        //
        // The cinematic pass-2 alpha-blended sprite mux is
        // 0xFFFFF238_FC127FFF — color = TEXEL0*SHADE, alpha = TEXEL0_A,
        // cycle1 = COMBINED passthrough. Even after particle/texrect-alpha
        // fixes the explosion sprites are not visible. To confirm whether
        // pipeline reaches the displayed framebuffer at all, this debug
        // rewrites the mux to output PRIMITIVE color at alpha=ONE
        // unconditionally:
        //   - color cycle1 D (H bits 6-8) = 3 (PRIMITIVE)
        //   - alpha cycle1 D (H bits 0-2) = 6 (ONE)
        // If we then see solid PRIM-colored blobs where explosions should
        // be, the issue is alpha-test/texture/SHADE — the geometry IS
        // hitting the visible fb. If we still see nothing, it's
        // fb-routing (sprites land in 0x80695C00 which VI never displays).
        //
        // Default off. ROGUESQ_PARTICLE_VISIBLE_DEBUG=1 to enable.
        static const bool particle_visible_debug = []() {
            const char *e = std::getenv("ROGUESQ_PARTICLE_VISIBLE_DEBUG");
            return e && atoi(e) != 0;
        }();
        // Match any mux with low-half 0xFC127FFF — the cinematic pass-2 mux
        // appears in multiple H-variants (0xFFFFFE3F = 1CYCLE, 0xFFFFF238 =
        // 2CYCLE). Both produce TEXEL0*SHADE color / TEXEL0_A alpha, so the
        // PRIM-opaque rewrite is appropriate for both.
        // We ALSO rewrite color cycle 0 to output PRIM directly so 1CYCLE muxes
        // (which use cycle 0 as final output) get the override too.
        // 0xFC11E623 is the dominant cinematic mux post-texrect-alpha-fix
        // (original 0xFC119623 → bits 12-14 SHADE→ONE → 0xFC11E623). Geo
        // trace confirmed it accounts for ~64% of cinematic TEXRECTs.
        if (particle_visible_debug && ((combine & 0xFFFFFFFFULL) == 0xFC127FFFULL ||
                                       (combine & 0xFFFFFFFFULL) == 0xFC11FE23ULL ||
                                       (combine & 0xFFFFFFFFULL) == 0xFC11E623ULL ||
                                       (combine & 0xFFFFFFFFULL) == 0xFC119623ULL)) {
            uint64_t old = combine;
            uint64_t L = combine & 0xFFFFFFFFULL;
            uint64_t H = (combine >> 32ULL) & 0xFFFFFFFFULL;
            // color cycle 0 D = 3 (PRIMITIVE) at H bits 15-17 (per parseColorInputD)
            H = (H & ~(0x7ULL << 15)) | (3ULL << 15);
            // color cycle 1 D = 3 (PRIMITIVE) at H bits 6-8
            H = (H & ~(0x7ULL << 6)) | (3ULL << 6);
            // alpha cycle 0 D = 6 (ONE) at H bits 9-11 (alphaInputABD)
            H = (H & ~(0x7ULL << 9)) | (6ULL << 9);
            // alpha cycle 1 D = 6 (ONE) at H bits 0-2
            H = (H & ~(0x7ULL << 0)) | (6ULL << 0);
            combine = (L) | (H << 32);
            if (combine != old) {
                static int n = 0;
                if (++n <= 5) {
                    fprintf(stderr, "[particle-visible-debug] rewrote 0x%016llX -> 0x%016llX #%d\n",
                        (unsigned long long)old, (unsigned long long)combine, n);
                    fflush(stderr);
                }
            }
        }
        // [trace] setCombine logging — every fresh mux + every 200th setCombine.
        // High-volume during cinematic (~60-100 lines/sec). Useful when
        // identifying which combiner is active for a specific draw, or
        // discovering new muxes the game uses. Default off.
        // ROGUESQ_LOG_RDP_STATE=1 (or ROGUESQ_LOG_ALL=1).
        static const bool log_rdp = []{
            const char *a = std::getenv("ROGUESQ_LOG_ALL");
            if (a && *a && *a != '0') return true;
            const char *e = std::getenv("ROGUESQ_LOG_RDP_STATE");
            return e && *e && *e != '0';
        }();
        { static int n=0; static uint64_t last = ~uint64_t(0);
            static std::unordered_set<uint64_t> s_uniq;
            if (log_rdp && combine != last) {
                ++n;
                bool fresh = s_uniq.insert(combine).second;
                if (fresh || n <= 30 || (n % 200) == 0) {
                    interop::ColorCombiner cc;
                    cc.L = combine & 0xFFFFFFFFULL;
                    cc.H = (combine >> 32ULL) & 0xFFFFFFFFULL;
                    // Decode both cycles' alpha inputs so we can spot a
                    // permanently-zero alpha (which would explain invisible
                    // glyphs in the credits/post-credits text path).
                    const uint32_t cyc = (uint32_t)otherMode.cycleType();
                    const char *cycName =
                        (cyc == G_CYC_1CYCLE) ? "1CYCLE" :
                        (cyc == G_CYC_2CYCLE) ? "2CYCLE" :
                        (cyc == G_CYC_COPY) ? "COPY" :
                        (cyc == G_CYC_FILL) ? "FILL" : "?";
                    fprintf(stderr, "[trace] setCombine #%d mux=0x%016llX L=0x%08X H=0x%08X cycleType=%s(%u)\n",
                        n, (unsigned long long)combine, cc.L, cc.H, cycName, cyc);
                    // For the particle mux specifically, dump the full
                    // otherMode + blender breakdown so we can see what
                    // alpha/coverage semantics the game expects.
                    //
                    // Trigger conditions (any one):
                    //   1. Specific muxes we've previously chased (kept for parity).
                    //   2. Blender is in "alpha-over-fb" form — cycle1 sources are
                    //      A = CC_ALPHA (0) and M = FB (1). Captures the cinematic
                    //      pass-2 sprite/particle path (L bits 0xC811xxxx) and any
                    //      future transparent-draw mux without us hard-coding it.
                    const uint32_t L_check = otherMode.L;
                    const uint32_t blender_check = (L_check >> 16) & 0xFFFF;
                    const bool isAlphaOverFb =
                        ((blender_check & 0x0300) == 0) &&            // cycle1 A = CC_ALPHA
                        ((blender_check & 0x0030) == 0x0010);          // cycle1 M = FB
                    if (combine == 0xFFFFFE3FFC127FFFULL ||
                        combine == 0xFFFFF238FC127FFFULL ||
                        isAlphaOverFb) {
                        const uint32_t L = otherMode.L;
                        const uint32_t H = otherMode.H;
                        const uint32_t blenderInputs = (L >> 16) & 0xFFFF;
                        fprintf(stderr,
                            "  [particle-mux] otherMode H=0x%08X L=0x%08X\n"
                            "    cvgXAlpha=%u alphaCvgSel=%u forceBlend=%u clrOnCvg=%u cvgDst=%u\n"
                            "    blenderInputs=0x%04X (bits L[31:16])\n"
                            "    cycle0: P=%u M=%u A=%u B=%u\n"
                            "    cycle1: P=%u M=%u A=%u B=%u\n",
                            H, L,
                            (unsigned)otherMode.cvgXAlpha(),
                            (unsigned)otherMode.alphaCvgSel(),
                            (unsigned)otherMode.forceBlend(),
                            (unsigned)otherMode.clrOnCvg(),
                            (unsigned)otherMode.cvgDst(),
                            blenderInputs,
                            (blenderInputs >> 14) & 0x3,
                            (blenderInputs >>  6) & 0x3,
                            (blenderInputs >> 10) & 0x3,
                            (blenderInputs >>  2) & 0x3,
                            (blenderInputs >> 12) & 0x3,
                            (blenderInputs >>  4) & 0x3,
                            (blenderInputs >>  8) & 0x3,
                            (blenderInputs >>  0) & 0x3);
                        fflush(stderr);
                    }
                    fprintf(stderr, "  cycle0 alpha A=%u B=%u C=%u D=%u\n",
                        cc.parseAlphaInputA(false), cc.parseAlphaInputB(false),
                        cc.parseAlphaInputC(false), cc.parseAlphaInputD(false));
                    fprintf(stderr, "  cycle1 alpha A=%u B=%u C=%u D=%u\n",
                        cc.parseAlphaInputA(true), cc.parseAlphaInputB(true),
                        cc.parseAlphaInputC(true), cc.parseAlphaInputD(true));
                    fprintf(stderr, "  cycle0 color A=%u B=%u C=%u D=%u\n",
                        cc.parseColorInputA(false), cc.parseColorInputB(false),
                        cc.parseColorInputC(false), cc.parseColorInputD(false));
                    fprintf(stderr, "  cycle1 color A=%u B=%u C=%u D=%u\n",
                        cc.parseColorInputA(true), cc.parseColorInputB(true),
                        cc.parseColorInputC(true), cc.parseColorInputD(true));
                    fflush(stderr);
                }
                last = combine;
            }
        }
        interop::ColorCombiner &colorCombiner = colorCombinerStack[colorCombinerStackSize - 1];
        colorCombiner.L = combine & 0xFFFFFFFFULL;
        colorCombiner.H = (combine >> 32ULL) & 0xFFFFFFFFULL;
        state->updateDrawStatusAttribute(DrawAttribute::Combine);
    }

    void RDP::pushCombine() {
        if (colorCombinerStackSize < RDP_EXTENDED_STACK_SIZE) {
            colorCombinerStack[colorCombinerStackSize] = colorCombinerStack[colorCombinerStackSize - 1];
            colorCombinerStackSize++;
        }
    }

    void RDP::popCombine() {
        if (colorCombinerStackSize > 1) {
            colorCombinerStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::Combine);
        }
    }

    void RDP::setTile(uint8_t tile, uint8_t fmt, uint8_t siz, uint16_t line, uint16_t tmem, uint8_t palette, uint8_t cmt, uint8_t cms, uint8_t maskt, uint8_t masks, uint8_t shiftt, uint8_t shifts) {
#ifdef LOG_TILE_METHODS
        RT64_LOG_PRINTF("RDP::setTile(tile %u, fmt %u, siz %u, line %u, tmem %u, palette %u, cmt %u, cms %u, maskt %u, masks %u, shiftt %u, shifts %u)", tile, fmt, siz, line, tmem, palette, cmt, cms, maskt, masks, shiftt, shifts);
#endif

        assert(tile < RDP_TILES);

        auto &t = tiles[tile];
        t.fmt = fmt;
        t.siz = siz;
        t.line = line;
        t.tmem = tmem;
        t.palette = palette;
        t.cmt = cmt;
        t.cms = cms;
        t.masks = masks;
        t.maskt = maskt;
        t.shifts = shifts;
        t.shiftt = shiftt;
        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }

    void RDP::setTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
#ifdef LOG_TILE_METHODS
        RT64_LOG_PRINTF("RDP::setTileSize(tile %u, uls %u, ult %u, lrs %u, lrt %u)", tile, uls, ult, lrs, lrt);
#endif

        assert(tile < RDP_TILES);
        auto &t = tiles[tile];
        t.uls = uls;
        t.ult = ult;
        t.lrs = lrs;
        t.lrt = lrt;
        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }

    void RDP::clearTileReplacementHash(uint8_t tile) {
        assert(tile < RDP_TILES);
        tileReplacementHashes[tile] = 0;
        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }

    void RDP::setTileReplacementHash(uint8_t tile, uint64_t replacementHash) {
        assert(tile < RDP_TILES);
        tileReplacementHashes[tile] = replacementHash;
        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }
    
    template<bool RGBA32 = false, bool TLUT = false>
    __forceinline void loadWord(uint8_t *TMEM, uint32_t tmemAddress, uint32_t tmemXorMask, const uint8_t *RDRAM, uint32_t textureAddress) {
        // Only sample the first two bytes in TLUT mode.
        uint32_t offsetMask;
        if constexpr (TLUT) {
            offsetMask = 0x1;
        }
        else {
            offsetMask = 0x7;
        }

        if constexpr (RGBA32) {
            // Split the lower and upper half of the word into the lower and upper half of TMEM.
            const uint32_t UpperTMEM = (RDP_TMEM_BYTES >> 1);
            TMEM[(tmemAddress + 0) ^ tmemXorMask] = RDRAM[(textureAddress + (0 & offsetMask)) ^ 3];
            TMEM[(tmemAddress + 1) ^ tmemXorMask] = RDRAM[(textureAddress + (1 & offsetMask)) ^ 3];
            TMEM[(tmemAddress + 2) ^ tmemXorMask] = RDRAM[(textureAddress + (4 & offsetMask)) ^ 3];
            TMEM[(tmemAddress + 3) ^ tmemXorMask] = RDRAM[(textureAddress + (5 & offsetMask)) ^ 3];
            TMEM[((tmemAddress + 0) ^ tmemXorMask) | UpperTMEM] = RDRAM[(textureAddress + (2 & offsetMask)) ^ 3];
            TMEM[((tmemAddress + 1) ^ tmemXorMask) | UpperTMEM] = RDRAM[(textureAddress + (3 & offsetMask)) ^ 3];
            TMEM[((tmemAddress + 2) ^ tmemXorMask) | UpperTMEM] = RDRAM[(textureAddress + (6 & offsetMask)) ^ 3];
            TMEM[((tmemAddress + 3) ^ tmemXorMask) | UpperTMEM] = RDRAM[(textureAddress + (7 & offsetMask)) ^ 3];
        }
        else {
            // Copy the entire word.
            for (uint32_t i = 0; i < 8; i++) {
                TMEM[(tmemAddress + i) ^ tmemXorMask] = RDRAM[(textureAddress + (i & offsetMask)) ^ 3];
            }
        }
    }

    template<bool RGBA32 = false, bool BLOCK = false, bool TLUT = false>
    __forceinline void loadToTMEMCommon(uint8_t *TMEM, const uint8_t *RDRAM, uint32_t textureStart, uint32_t textureStride, uint32_t tmemStart,
        uint32_t tmemStride, uint32_t wordsPerRow, uint32_t rowCount, uint32_t dxtIncrement = 0)
    {
        assert((!BLOCK || (rowCount == 1)) && "Load block must behave as if it only loads one row of data.");
        
        const uint32_t DXTSwap = 0x800;
        uint32_t textureAddress, tmemAddress, wordCount, tmemMask, tmemAdvance;
        if constexpr (RGBA32) {
            tmemMask = RDP_TMEM_MASK16;
            tmemAdvance = 0x4;
        }
        else {
            tmemMask = RDP_TMEM_MASK8;
            tmemAdvance = 0x8;
        }

        uint32_t textureAdvance;
        if constexpr (TLUT) {
            textureAdvance = 0x2;
        }
        else {
            textureAdvance = 0x8;
        }

        uint32_t tmemXorMask = 0x0;
        uint32_t dxtCounter = 0x0;
        auto loadWordStep = [&]() {
            if constexpr (BLOCK) {
                dxtCounter += dxtIncrement;
                while (dxtCounter >= DXTSwap) {
                    tmemAddress = (tmemAddress + tmemStride) & tmemMask;
                    dxtCounter -= DXTSwap;
                    tmemXorMask ^= 0x4;
                }
            }

            textureAddress += textureAdvance;
            tmemAddress = (tmemAddress + tmemAdvance) & tmemMask;
        };
        
        uint32_t textureAddressRow = textureStart;
        uint32_t tmemAddressRow = tmemStart & tmemMask;
        auto loadRowStep = [&]() {
            tmemAddressRow = (tmemAddressRow + tmemStride) & tmemMask;
            textureAddressRow += textureStride;
            tmemXorMask ^= 0x4;
        };

        while (rowCount > 0) {
            textureAddress = textureAddressRow;
            tmemAddress = tmemAddressRow;
            wordCount = wordsPerRow;
            while (wordCount > 0) {
                loadWord<RGBA32, TLUT>(TMEM, tmemAddress, tmemXorMask, RDRAM, textureAddress);
                loadWordStep();
                wordCount--;
            }

            loadRowStep();
            rowCount--;
        }
    }

    void RDP::loadTileOperation(const LoadTile &loadTile, const LoadTexture &loadTexture, bool deferred) {
        const uint32_t bytesOffset = (loadTile.uls >> 2) << loadTexture.siz >> 1;
        const uint32_t bytesPerRow = loadTexture.width << loadTexture.siz >> 1;
        const uint32_t textureStart = loadTexture.address + bytesOffset + bytesPerRow * (loadTile.ult >> 2);
        const uint32_t rowCount = 1 + ((loadTile.lrt >> 2) - (loadTile.ult >> 2));
        const uint32_t tileWidth = ((loadTile.lrs >> 2) - (loadTile.uls >> 2));
        const uint32_t wordsPerRow = (tileWidth >> (4 - loadTile.siz)) + 1;
        const uint32_t tmemStart = loadTile.tmem << 3;
        const uint32_t tmemStride = loadTile.line << 3;
        const uint32_t textureEnd = textureStart + (rowCount - 1) * bytesPerRow + (wordsPerRow << 3);
        const bool RGBA32 = (loadTile.siz == G_IM_SIZ_32b) && (loadTile.fmt == G_IM_FMT_RGBA);
        if (deferred) {
            checkImageOverlap(textureStart, textureEnd);

            // Discard any FB regions currently loaded into TMEM within the specified range
            const uint32_t wordShift = RGBA32 ? 2 : 3;
            const uint32_t tmemMask = RGBA32 ? RDP_TMEM_MASK128 : RDP_TMEM_MASK64;
            const uint32_t tmemBytes = (rowCount - 1) * tmemStride + (wordsPerRow << wordShift);
            const uint32_t tmemEnd = tmemStart + tmemBytes;
            state->framebufferManager.discardRegionsTMEM(tmemStart >> 3, tmemBytes >> 3, tmemMask);

            // Check for any GPU copies that can be made.
            const uint32_t lineShift = RGBA32 ? 1 : 0;
            const uint32_t lineWidth = loadTile.line << ((4 + lineShift) - loadTile.siz);
            checkFramebufferOverlap(tmemStart >> 3, tmemBytes >> 3, tmemMask, textureStart, textureEnd, lineWidth, rowCount, RGBA32, true);
        }
        else {
            // ROGUESQ defensive end-address check. The existing pre-fullSync
            // check rejects load ops with start address >= 8MB, but doesn't
            // check the END address. loadToTMEMCommon reads
            // (rowCount-1)*bytesPerRow + (wordsPerRow << 3) bytes from
            // textureStart, and SEH-AVs if that goes past RDRAM. Skip
            // here too rather than crash gfx_thread.
            constexpr uint32_t kRdramSize = 0x800000;
            if (textureEnd > kRdramSize ||
                textureStart >= kRdramSize ||
                rowCount == 0 || rowCount > 1024 ||
                wordsPerRow == 0 || wordsPerRow > 4096 ||
                bytesPerRow == 0 || bytesPerRow > 0x10000) {
                static std::atomic<uint64_t> s_skipped{0};
                uint64_t n = ++s_skipped;
                if (n == 1 || (n & (n - 1)) == 0) {
                    fprintf(stderr,
                        "[rt64] skip loadTile OOB #%llu start=0x%X end=0x%X rows=%u words/row=%u bytes/row=%u\n",
                        (unsigned long long)n,
                        textureStart, textureEnd, rowCount, wordsPerRow, bytesPerRow);
                    fflush(stderr);
                }
                return;
            }
            // Load into TMEM.
            uint8_t *TMEM8 = reinterpret_cast<uint8_t *>(TMEM);
            const uint8_t *RDRAM = state->RDRAM;
            if (RGBA32) {
                loadToTMEMCommon<true>(TMEM8, RDRAM, textureStart, bytesPerRow, tmemStart, tmemStride, wordsPerRow, rowCount);
            }
            else {
                loadToTMEMCommon<false>(TMEM8, RDRAM, textureStart, bytesPerRow, tmemStart, tmemStride, wordsPerRow, rowCount);
            }
        }
    }

    void RDP::loadBlockOperation(const LoadTile &loadTile, const LoadTexture &loadTexture, bool deferred) {
        // Deduce loading parameters for block and the texture address.
        const uint32_t bytesOffset = loadTile.uls << loadTexture.siz >> 1;
        const uint32_t bytesPerRow = loadTexture.width << loadTexture.siz >> 1;
        const uint32_t textureStart = loadTexture.address + bytesOffset + bytesPerRow * loadTile.ult;
        const uint32_t wordCount = ((loadTile.lrs - loadTile.uls) >> (4 - loadTile.siz)) + 1;
        const uint32_t tmemStart = loadTile.tmem << 3;
        const uint32_t tmemStride = loadTile.line << 3;
        const uint32_t textureEnd = textureStart + (wordCount << 3);
        const bool RGBA32 = (loadTile.siz == G_IM_SIZ_32b) && (loadTile.fmt == G_IM_FMT_RGBA);
        if (deferred) {
            checkImageOverlap(textureStart, textureEnd);

            // Discard any FB regions currently loaded into TMEM within the specified range.
            const uint32_t wordShift = RGBA32 ? 2 : 3;
            const uint32_t tmemMask = RGBA32 ? RDP_TMEM_MASK128 : RDP_TMEM_MASK64;
            const uint32_t tmemBytes = (wordCount << wordShift);
            const uint32_t tmemEnd = tmemStart + tmemBytes;
            state->framebufferManager.discardRegionsTMEM(tmemStart >> 3, tmemBytes >> 3, tmemMask);

            // Check for any GPU copies that can be made.
            checkFramebufferOverlap(tmemStart >> 3, tmemBytes >> 3, tmemMask, textureStart, textureEnd, 0, 0, RGBA32, true);
        }
        else {
            // Load into TMEM.
            uint8_t *TMEM8 = reinterpret_cast<uint8_t *>(TMEM);
            const uint8_t *RDRAM = state->RDRAM;
            if (RGBA32) {
                loadToTMEMCommon<true, true>(TMEM8, RDRAM, textureStart, bytesPerRow, tmemStart, tmemStride, wordCount, 1, loadTile.lrt);
            }
            else {
                loadToTMEMCommon<false, true>(TMEM8, RDRAM, textureStart, bytesPerRow, tmemStart, tmemStride, wordCount, 1, loadTile.lrt);
            }
        }
    }

    void RDP::loadTLUTOperation(const LoadTile &loadTile, const LoadTexture &loadTexture, bool deferred) {
        // Deduce loading parameters for TLUT and the texture address.
        const uint32_t bytesOffset = (loadTile.uls >> 2) << loadTexture.siz >> 1;
        const uint32_t bytesPerRow = loadTexture.width << loadTexture.siz >> 1;
        const uint32_t textureStart = loadTexture.address + bytesOffset + bytesPerRow * (loadTile.ult >> 2);
        const uint32_t rowCount = 1 + ((loadTile.lrt >> 2) - (loadTile.ult >> 2));
        const uint32_t wordsPerRow = ((loadTile.lrs >> 2) - (loadTile.uls >> 2)) + 1;
        const uint32_t tmemStart = loadTile.tmem << 3;
        const uint32_t tmemStride = loadTile.line << 5;
        const bool RGBA32 = (loadTile.siz == G_IM_SIZ_32b) && (loadTile.fmt == G_IM_FMT_RGBA);
        if (deferred) {
            // Flush current framebuffer pair if any of the images being written to are loaded by this TLUT.
            const uint32_t textureEnd = textureStart + (rowCount - 1) * bytesPerRow + (wordsPerRow << 3);
            checkImageOverlap(textureStart, textureEnd);

            // Discard any FB regions currently loaded into TMEM within the specified range.
            const uint32_t wordShift = RGBA32 ? 0 : 1;
            const uint32_t tmemMask = RGBA32 ? RDP_TMEM_MASK128 : RDP_TMEM_MASK64;
            const uint32_t tmemBytes = (rowCount - 1) * tmemStride + (wordsPerRow << wordShift);
            const uint32_t tmemEnd = tmemStart + tmemBytes;
            state->framebufferManager.discardRegionsTMEM(tmemStart >> 3, tmemBytes >> 3, tmemMask);

            // Mark the TMEM regions with the loaded TLUT.
            const uint32_t lineShift = RGBA32 ? 1 : 0;
            const uint32_t tileWidth = loadTile.line << ((4 + lineShift) - loadTile.siz);
            checkFramebufferOverlap(tmemStart >> 3, tmemBytes >> 3, tmemMask, textureStart, textureEnd, 0, 0, RGBA32, false);
        }
        else {
            // Load into TMEM.
            uint8_t *TMEM8 = reinterpret_cast<uint8_t *>(TMEM);
            const uint8_t *RDRAM = state->RDRAM;
            // Bounds-check the texture source. With a Factor5-driven game, an
            // upstream SETTIMG sometimes carries a wildly-out-of-range address
            // (observed: 28GB-shaped values), AVing inside loadToTMEMCommon. Skip
            // the load entirely and log so we know we're losing TLUTs.
            constexpr uint32_t kRDRAMSize = 8 * 1024 * 1024;
            const uint32_t textureEnd = textureStart + (rowCount - 1) * bytesPerRow + (wordsPerRow << 3);
            if (textureStart >= kRDRAMSize || textureEnd > kRDRAMSize || textureEnd < textureStart) {
                static int n = 0;
                if (++n <= 10 || (n % 500) == 0) {
                    fprintf(stderr, "[loadTLUT] skip out-of-range #%d textureStart=0x%08X textureEnd=0x%08X\n",
                        n, textureStart, textureEnd);
                    fflush(stderr);
                }
                return;
            }
            if (RGBA32) {
                loadToTMEMCommon<true, false, true>(TMEM8, RDRAM, textureStart, bytesPerRow, tmemStart, tmemStride, wordsPerRow, rowCount);
            }
            else {
                loadToTMEMCommon<false, false, true>(TMEM8, RDRAM, textureStart, bytesPerRow, tmemStart, tmemStride, wordsPerRow, rowCount);
            }
        }
    }

    void RDP::loadTile(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
#ifdef LOG_LOAD_METHODS
        RT64_LOG_PRINTF("RDP::loadTile(tile %u, uls %u, ult %u, lrs %u, lrt %u)", tile, uls, ult, lrs, lrt);
#endif
        assert(tile < RDP_TILES);
        auto &t = tiles[tile];
        t.uls = uls;
        t.ult = ult;
        t.lrs = lrs;
        t.lrt = lrt;

        // Ignored by the hardware.
        if (t.uls > t.lrs) {
            return;
        }

#   ifdef ASSERT_LOAD_METHODS
        assert((t.uls != t.lrs) || ((t.uls & 0x3) == 0) && "Unknown and possibly undefined hardware behavior.");
        assert((t.ult != t.lrt) || ((t.ult & 0x3) == 0) && "Unknown hardware behavior.");
        assert((t.lrt <= t.lrt) && "Unknown hardware behavior.");
        assert((t.siz == texture.siz) && "Different tile and texture sizes are not currently supported.");
        assert((texture.siz != G_IM_SIZ_4b) && "4-bit texture image is not currently supported.");
        assert((t.fmt != G_IM_FMT_YUV) && "YUV is not currently supported.");
        assert(((t.siz != G_IM_SIZ_32b) || (t.fmt == G_IM_FMT_RGBA)) && "Other 32-bit formats than RGBA32 are not currently supported.");
#   endif

        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const bool warningsEnabled = state->ext.userConfig->developerMode;
        if (warningsEnabled) {
            const uint32_t loadIndex = uint32_t(workload.drawData.loadOperations.size());
            if (t.siz != texture.siz) {
                CommandWarning warning = CommandWarning::format("Load Operation #%u: RDP::loadTile called with texture image siz %u and "
                    "tile descriptor #%u with siz %u. Pixel size mismatch might not work correctly.", loadIndex, texture.siz, tile, t.siz);

                warning.indexType = CommandWarning::IndexType::LoadIndex;
                warning.load.index = loadIndex;
                workload.commandWarnings.emplace_back(warning);
            }
        }

        // Perform the first step of the deferred operation.
        loadTileOperation(t, texture, true);

        // Store the operation.
        LoadOperation operation;
        operation.type = LoadOperation::Type::Tile;
        operation.tile = t;
        operation.texture = texture;
        auto &opTile = operation.operationTile;
        opTile.tile = tile;
        opTile.uls = uls;
        opTile.ult = ult;
        opTile.lrs = lrs;
        opTile.lrt = lrt;
        workload.drawData.loadOperations.emplace_back(operation);

        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }

    bool RDP::loadTileCopyCheck(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
        assert(tile < RDP_TILES);

        // Ignored by the hardware.
        if (uls > lrs) {
            return false;
        }

        const bool gpuCopiesEnabled = state->ext.emulatorConfig->framebuffer.copyWithGPU;
        if (!gpuCopiesEnabled) {
            return false;
        }

        // Current framebuffer pair must be flushed if we wish to do a tile copy with any of the bytes it uses.
        const auto &t = tiles[tile];
        const uint32_t bytesOffset = (uls >> 2) << texture.siz >> 1;
        const uint32_t bytesPerRow = texture.width << texture.siz >> 1;
        const uint32_t textureStart = texture.address + bytesOffset + bytesPerRow * (ult >> 2);
        const uint32_t rowCount = 1 + ((lrt >> 2) - (ult >> 2));
        const uint32_t tileWidth = ((lrs >> 2) - (uls >> 2));
        const uint32_t wordsPerRow = (tileWidth >> (4 - t.siz)) + 1;
        const uint32_t textureSize = (rowCount - 1) * bytesPerRow + (wordsPerRow << 3);
        const uint32_t textureEnd = textureStart + textureSize;
        checkImageOverlap(textureStart, textureEnd);

        Framebuffer *framebuffer = state->framebufferManager.findMostRecentContaining(textureStart, textureEnd);
        if (framebuffer != nullptr) {
            FramebufferTile fbTile;
            const bool RGBA32 = (t.siz == G_IM_SIZ_32b) && (t.fmt == G_IM_FMT_RGBA);
            const uint32_t lineShift = RGBA32 ? 1 : 0;
            const uint32_t lineWidth = t.line << ((4 + lineShift) - t.siz);
            return state->framebufferManager.makeFramebufferTile(framebuffer, textureStart, textureEnd, lineWidth, rowCount, fbTile, RGBA32);
        }
        else {
            return false;
        }
    }

    bool RDP::loadTileReplacementCheck(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt, uint8_t imageSiz, uint8_t imageFmt, uint16_t imageLoad, uint16_t imagePal, uint64_t &replacementHash) {
        assert(tile < RDP_TILES);

        // Ignored by the hardware.
        if (uls > lrs) {
            return false;
        }

        const auto &t = tiles[tile];
        const uint32_t bytesOffset = (uls >> 2) << texture.siz >> 1;
        const uint32_t bytesPerRow = texture.width << texture.siz >> 1;
        const uint32_t textureStart = texture.address + bytesOffset + bytesPerRow * (ult >> 2);
        const uint32_t rowCount = 1 + ((lrt >> 2) - (ult >> 2));
        const uint32_t tileWidth = ((lrs >> 2) - (uls >> 2));
        const uint32_t wordsPerRow = (tileWidth >> (4 - t.siz)) + 1;
        const uint32_t textureSize = (rowCount - 1) * bytesPerRow + (wordsPerRow << 3);

        // Hash the entire texture's memory along with the parameters that affect its format.
        XXH3_state_t xxh3;
        XXH3_64bits_reset(&xxh3);
        XXH3_64bits_update(&xxh3, &state->RDRAM[textureStart], textureSize);
        XXH3_64bits_update(&xxh3, &imageSiz, sizeof(imageSiz));
        XXH3_64bits_update(&xxh3, &imageFmt, sizeof(imageFmt));
        XXH3_64bits_update(&xxh3, &imageLoad, sizeof(imageLoad));
        XXH3_64bits_update(&xxh3, &imagePal, sizeof(imagePal));
        replacementHash = XXH3_64bits_digest(&xxh3);

        // Return whether a replacement exists for this hash or not.
        return state->ext.textureCache->hasReplacement(replacementHash);
    }

    void RDP::loadBlock(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t dxt) {
#ifdef LOG_LOAD_METHODS
        RT64_LOG_PRINTF("RDP::loadBlock(tile %u, uls %u, ult %u, lrs %u, dxt %u)", tile, uls, ult, lrs, dxt);
#endif
        assert(tile < RDP_TILES);
        auto &t = tiles[tile];
        t.uls = uls;
        t.ult = ult;
        t.lrs = lrs;
        t.lrt = dxt;

        // Ignored by the hardware.
        if ((t.uls > t.lrs) || (t.lrs >= 0x800)) {
            return;
        }
        
#   ifdef ASSERT_LOAD_METHODS
        assert((t.uls != t.lrs) || ((t.uls & 0x3) == 0) && "Unknown and possibly undefined hardware behavior.");
        assert((t.siz == texture.siz) && "Different tile and texture sizes are not currently supported.");
        assert((texture.siz != G_IM_SIZ_4b) && "4-bit texture image is not currently supported.");
        assert((t.fmt != G_IM_FMT_YUV) && "YUV is not currently supported.");
        assert(((t.siz != G_IM_SIZ_32b) || (t.fmt == G_IM_FMT_RGBA)) && "Other 32-bit formats than RGBA32 are not currently supported.");
#   endif

        // Check for warnings in developer mode.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const bool warningsEnabled = state->ext.userConfig->developerMode;
        if (warningsEnabled) {
            const uint32_t loadIndex = uint32_t(workload.drawData.loadOperations.size());
            if (t.siz != texture.siz) {
                CommandWarning warning = CommandWarning::format("Load Operation #%u: RDP::loadBlock called with texture image siz %u and "
                    "tile descriptor #%u with siz %u. Pixel size mismatch might not work correctly.", loadIndex, texture.siz, tile, t.siz);

                warning.indexType = CommandWarning::IndexType::LoadIndex;
                warning.load.index = loadIndex;
                workload.commandWarnings.emplace_back(warning);
            }
        }

        // Perform the first step of the deferred operation.
        loadBlockOperation(t, texture, true);

        // Store the operation.
        LoadOperation operation;
        operation.type = LoadOperation::Type::Block;
        operation.tile = t;
        operation.texture = texture;
        auto &opBlock = operation.operationBlock;
        opBlock.tile = tile;
        opBlock.uls = uls;
        opBlock.ult = ult;
        opBlock.lrs = lrs;
        opBlock.dxt = dxt;
        workload.drawData.loadOperations.emplace_back(operation);

        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }

    void RDP::loadTLUT(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
#ifdef LOG_LOAD_METHODS
        RT64_LOG_PRINTF("RDP::loadTLUT(tile %u, uls %u, ult %u, lrs %u, lrt %u)", tile, uls, ult, lrs, lrt);
#endif
        assert(tile < RDP_TILES);
        auto &t = tiles[tile];
        t.uls = uls;
        t.ult = ult;
        t.lrs = lrs;
        t.lrt = lrt;

#   ifdef ASSERT_LOAD_METHODS
        assert((texture.siz == G_IM_SIZ_16b) && "Non-16 bit textures are not currently supported.");
        assert((t.siz == G_IM_SIZ_4b) && "Non-4 bit tiles are not currently supported.");
#   endif
        
        // Check for warnings in developer mode.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const bool warningsEnabled = state->ext.userConfig->developerMode;
        if (warningsEnabled) {
            const uint32_t loadIndex = uint32_t(workload.drawData.loadOperations.size());
            if (texture.siz != G_IM_SIZ_16b) {
                CommandWarning warning = CommandWarning::format("Load Operation #%u: RDP::loadTLUT called with incorrect texture image siz %u. "
                    "Loading TLUTs that don't use 16-bit formats might not work correctly.", loadIndex, texture.siz);

                warning.indexType = CommandWarning::IndexType::LoadIndex;
                warning.load.index = loadIndex;
                workload.commandWarnings.emplace_back(warning);
            }
            
            if (t.siz != G_IM_SIZ_4b) {
                CommandWarning warning = CommandWarning::format("Load Operation #%u: RDP::loadTLUT called with incorrect tile descriptor #%u with siz %u. "
                    "Loading TLUTs that don't use 4-bit formats as their tile size might not work correctly.", loadIndex, tile, t.siz);

                warning.indexType = CommandWarning::IndexType::LoadIndex;
                warning.load.index = loadIndex;
                workload.commandWarnings.emplace_back(warning);
            }
        }

        // Perform the first step of the deferred operation.
        loadTLUTOperation(t, texture, true);

        // Store the operation.
        LoadOperation operation;
        operation.type = LoadOperation::Type::TLUT;
        operation.tile = t;
        operation.texture = texture;
        auto &opTLUT = operation.operationTLUT;
        opTLUT.tile = tile;
        opTLUT.uls = uls;
        opTLUT.ult = ult;
        opTLUT.lrs = lrs;
        opTLUT.lrt = lrt;
        workload.drawData.loadOperations.emplace_back(operation);

        state->updateDrawStatusAttribute(DrawAttribute::Texture);
    }

    void RDP::setEnvColor(uint32_t color) {
        hlslpp::float4 &envColor = envColorStack[envColorStackSize - 1];
        envColor.x = ((color >> 24) & 0xFF) / 255.0f;
        envColor.y = ((color >> 16) & 0xFF) / 255.0f;
        envColor.z = ((color >> 8) & 0xFF) / 255.0f;
        envColor.w = ((color >> 0) & 0xFF) / 255.0f;
        state->updateDrawStatusAttribute(DrawAttribute::EnvColor);
    }

    void RDP::pushEnvColor() {
        if (envColorStackSize < RDP_EXTENDED_STACK_SIZE) {
            envColorStack[envColorStackSize] = envColorStack[envColorStackSize - 1];
            envColorStackSize++;
        }
    }

    void RDP::popEnvColor() {
        if (envColorStackSize > 1) {
            envColorStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::EnvColor);
        }
    }
    
    void RDP::setPrimColor(uint8_t lodFrac, uint8_t lodMin, uint32_t color) {
        { static int n=0;
          ++n;
          // ALWAYS log first 30, then every 1000th regardless of value change
          if (n <= 30 || (n % 1000) == 0) {
              if(false) fprintf(stderr, "[trace] setPrimColor #%d color=0x%08X A=%u\n",
                  n, color, (unsigned)(color & 0xFF));
              fflush(stderr);
          }
        }
        hlslpp::float2 &primLOD = primLODStack[primColorStackSize - 1];
        primLOD.x = lodFrac / 256.0f;
        primLOD.y = lodMin / 32.0f;

        hlslpp::float4 &primColor = primColorStack[primColorStackSize - 1];
        primColor.x = ((color >> 24) & 0xFF) / 255.0f;
        primColor.y = ((color >> 16) & 0xFF) / 255.0f;
        primColor.z = ((color >> 8) & 0xFF) / 255.0f;
        primColor.w = ((color >> 0) & 0xFF) / 255.0f;
        state->updateDrawStatusAttribute(DrawAttribute::PrimColor);
    }

    void RDP::pushPrimColor() {
        if (primColorStackSize < RDP_EXTENDED_STACK_SIZE) {
            primColorStack[primColorStackSize] = primColorStack[primColorStackSize - 1];
            primLODStack[primColorStackSize] = primLODStack[primColorStackSize - 1];
            primColorStackSize++;
        }
    }

    void RDP::popPrimColor() {
        if (primColorStackSize > 1) {
            primColorStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::PrimColor);
        }
    }

    void RDP::setBlendColor(uint32_t color) {
        hlslpp::float4 &blendColor = blendColorStack[blendColorStackSize - 1];
        blendColor.x = ((color >> 24) & 0xFF) / 255.0f;
        blendColor.y = ((color >> 16) & 0xFF) / 255.0f;
        blendColor.z = ((color >> 8) & 0xFF) / 255.0f;
        blendColor.w = ((color >> 0) & 0xFF) / 255.0f;
        state->updateDrawStatusAttribute(DrawAttribute::BlendColor);
    }

    void RDP::pushBlendColor() {
        if (blendColorStackSize < RDP_EXTENDED_STACK_SIZE) {
            blendColorStack[blendColorStackSize] = blendColorStack[blendColorStackSize - 1];
            blendColorStackSize++;
        }
    }

    void RDP::popBlendColor() {
        if (blendColorStackSize > 1) {
            blendColorStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::BlendColor);
        }
    }

    void RDP::setFogColor(uint32_t color) {
        hlslpp::float4 &fogColor = fogColorStack[fogColorStackSize - 1];
        fogColor.x = ((color >> 24) & 0xFF) / 255.0f;
        fogColor.y = ((color >> 16) & 0xFF) / 255.0f;
        fogColor.z = ((color >> 8) & 0xFF) / 255.0f;
        fogColor.w = ((color >> 0) & 0xFF) / 255.0f;
        state->updateDrawStatusAttribute(DrawAttribute::FogColor);
    }

    void RDP::pushFogColor() {
        if (fogColorStackSize < RDP_EXTENDED_STACK_SIZE) {
            fogColorStack[fogColorStackSize] = fogColorStack[fogColorStackSize - 1];
            fogColorStackSize++;
        }
    }

    void RDP::popFogColor() {
        if (fogColorStackSize > 1) {
            fogColorStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::FogColor);
        }
    }

    void RDP::setFillColor(uint32_t color) {
        // ROGUESQ_FILLCOLOR_DEBUG — force every FILL_COLOR write to a fixed
        // 32-bit value. Useful for visualizing where FILL-mode draws (FILLRECT
        // / TEXRECT-in-FILL-cycle) land in the framebuffer: any pixel whose
        // color comes from FILL_COLOR will turn into the override.
        //
        //   ROGUESQ_FILLCOLOR_DEBUG=1            → red   (default 0xF801F801, 16-bit RGBA-5551 ×2)
        //   ROGUESQ_FILLCOLOR_DEBUG=0xRRGGBBAA   → custom 32-bit value
        //
        // Reveal use: the cinematic uses FILL-mode draws to paint its
        // background. Setting this to red shows the whole cutscene background
        // as red, with model geometry rendered opaquely on top.
        static const uint32_t fill_override = []() -> uint32_t {
            const char *e = std::getenv("ROGUESQ_FILLCOLOR_DEBUG");
            if (!e || !*e) return 0;
            // strtoul handles "0x..." and decimal; "1" maps to default red.
            uint32_t v = (uint32_t)strtoul(e, nullptr, 0);
            if (v == 0) return 0;            // disabled
            if (v == 1) return 0xF801F801u;  // canonical red 5551
            return v;
        }();
        if (fill_override != 0) {
            uint32_t old = color;
            color = fill_override;
            static int n = 0;
            if (++n <= 5) {
                fprintf(stderr, "[fillcolor-debug] rewrote 0x%08X -> 0x%08X #%d\n", old, color, n);
                fflush(stderr);
            }
        }
        fillColorStack[fillColorStackSize - 1] = color;
        state->updateDrawStatusAttribute(DrawAttribute::FillColor);
    }

    void RDP::pushFillColor() {
        if (fillColorStackSize < RDP_EXTENDED_STACK_SIZE) {
            fillColorStack[fillColorStackSize] = fillColorStack[fillColorStackSize - 1];
            fillColorStackSize++;
        }
    }

    void RDP::popFillColor() {
        if (fillColorStackSize > 1) {
            fillColorStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::FillColor);
        }
    }

    void RDP::setOtherMode(uint32_t high, uint32_t low) {
        // Log cycle-type transitions (1CYCLE / 2CYCLE / COPY / FILL).
        // Useful for correlating with the active combiner — in FILL mode
        // the combiner mux is "stored but not applied", which can confuse
        // mux-based diagnostics (see particle-fix history). One line per
        // cycle-type change; gated to ~30 transitions then every 200th.
        // Default off. ROGUESQ_LOG_RDP_STATE=1 (or ROGUESQ_LOG_ALL=1).
        static const bool log_rdp = []{
            const char *a = std::getenv("ROGUESQ_LOG_ALL");
            if (a && *a && *a != '0') return true;
            const char *e = std::getenv("ROGUESQ_LOG_RDP_STATE");
            return e && *e && *e != '0';
        }();
        if (log_rdp) { static uint32_t lastCyc = 0xFFFFFFFFu;
          uint32_t newCyc = (high >> G_MDSFT_CYCLETYPE) & 0x3;
          if (newCyc != lastCyc) {
              lastCyc = newCyc;
              static int n = 0;
              if (++n <= 30 || (n % 200) == 0) {
                  const char *name = (newCyc == G_CYC_1CYCLE) ? "1CYCLE"
                                   : (newCyc == G_CYC_2CYCLE) ? "2CYCLE"
                                   : (newCyc == G_CYC_COPY)   ? "COPY"
                                   : (newCyc == G_CYC_FILL)   ? "FILL" : "?";
                  fprintf(stderr, "[trace] setOtherMode #%d cycleType=%s(%u) H=0x%08X L=0x%08X\n",
                      n, name, newCyc, high, low);
                  fflush(stderr);
              }
          }
        }
        // ROGUESQ_DISABLE_Z_CMP — diagnostic. Forces Z_CMP and Z_UPD bits OFF
        // for every otherMode write. Tests the hypothesis that cinematic
        // TEXRECTs are being depth-rejected. If the explosion appears with
        // this knob on (but other geometry breaks), Z-test was killing the
        // sprites.
        static const bool disable_z = []() {
            const char *e = std::getenv("ROGUESQ_DISABLE_Z_CMP");
            return e && atoi(e) != 0;
        }();
        if (disable_z) {
            const uint32_t Z_CMP_BIT = 0x10;
            const uint32_t Z_UPD_BIT = 0x20;
            uint32_t newL = low & ~(Z_CMP_BIT | Z_UPD_BIT);
            if (newL != low) {
                static int n = 0;
                if (++n <= 5) {
                    fprintf(stderr, "[disable-z] otherMode L 0x%08X -> 0x%08X #%d\n", low, newL, n);
                    fflush(stderr);
                }
            }
            low = newL;
        }
        otherMode.H = high;
        otherMode.L = low;
        state->updateDrawStatusAttribute(DrawAttribute::OtherMode);
    }

    void RDP::setPrimDepth(uint16_t z, uint16_t dz) {
        const float Fixed15ToFloat = 1.0f / 32767.0f;
        const float Fixed16ToFloat = 1.0f / 65535.0f;
        hlslpp::float2 &primDepth = primDepthStack[primDepthStackSize - 1];
        primDepth.x = (z & 0x7FFFU) * Fixed15ToFloat;
        primDepth.y = (dz & 0xFFFFU) * Fixed16ToFloat;
        state->updateDrawStatusAttribute(DrawAttribute::PrimDepth);
    }

    void RDP::setScissor(uint8_t mode, int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
        setScissor(mode, ulx, uly, lrx, lry, extended.global.scissor);
    }
    
    void RDP::setScissor(uint8_t mode, int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, const ExtendedAlignment &extAlignment) {
        FixedRect &scissorRect = scissorRectStack[scissorStackSize - 1];
        scissorRect.ulx = std::clamp(movedFromOrigin(ulx + extAlignment.leftOffset, extAlignment.leftOrigin), extAlignment.leftBound, extAlignment.rightBound);
        scissorRect.uly = std::clamp(uly + extAlignment.topOffset, extAlignment.topBound, extAlignment.bottomBound);
        scissorRect.lrx = std::clamp(movedFromOrigin(lrx + extAlignment.rightOffset, extAlignment.rightOrigin), extAlignment.leftBound, extAlignment.rightBound);
        scissorRect.lry = std::clamp(lry + extAlignment.bottomOffset, extAlignment.topBound, extAlignment.bottomBound);
        scissorModeStack[scissorStackSize - 1] = mode;
        extended.scissorLeftOrigin = extAlignment.leftOrigin;
        extended.scissorRightOrigin = extAlignment.rightOrigin;
        state->updateDrawStatusAttribute(DrawAttribute::Scissor);
    }

    void RDP::pushScissor() {
        if (scissorStackSize < RDP_EXTENDED_STACK_SIZE) {
            scissorRectStack[scissorStackSize] = scissorRectStack[scissorStackSize - 1];
            scissorModeStack[scissorStackSize] = scissorModeStack[scissorStackSize - 1];
            scissorStackSize++;
        }
    }

    void RDP::popScissor() {
        if (scissorStackSize > 1) {
            scissorStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::Scissor);
        }
    }

    void RDP::setConvert(int32_t k0, int32_t k1, int32_t k2, int32_t k3, int32_t k4, int32_t k5) {
        convertK[0] = k0;
        convertK[1] = k1;
        convertK[2] = k2;
        convertK[3] = k3;
        convertK[4] = k4;
        convertK[5] = k5;
        state->updateDrawStatusAttribute(DrawAttribute::Convert);
    }

    void RDP::setKeyR(uint32_t cR, uint32_t sR, uint32_t wR) {
        // Width is ignored until its exact purpose is understood on the chroma keying process.
        keyCenter.x = cR / 255.0f;
        keyScale.x = sR / 255.0f;
        state->updateDrawStatusAttribute(DrawAttribute::Key);
    }

    void RDP::setKeyGB(uint32_t cG, uint32_t sG, uint32_t wG, uint32_t cB, uint32_t sB, uint32_t wB) {
        // Width is ignored until its exact purpose is understood on the chroma keying process.
        keyCenter.y = cG / 255.0f;
        keyCenter.z = cB / 255.0f;
        keyScale.y = sG / 255.0f;
        keyScale.z = sB / 255.0f;
        state->updateDrawStatusAttribute(DrawAttribute::Key);
    }

    void RDP::fillRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
        fillRect(ulx, uly, lrx, lry, extended.global.rect);
    }

    void RDP::fillRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, const ExtendedAlignment &extAlignment) {
#   ifdef LOG_FILLRECT_METHODS
        RT64_LOG_PRINTF("RDP::fillRect(ulx %d, uly %d, lrx %d, lry %d)", ulx, uly, lrx, lry);
#   endif

        // Filter out incorrect rectangles.
        if ((lrx < ulx) || (lry < uly)) {
            return;
        }

        int32_t mode = (otherMode.H & (3U << G_MDSFT_CYCLETYPE));
        if ((mode == G_CYC_COPY) || (mode == G_CYC_FILL)) {
            lrx |= 3;
            lry |= 3;
        }

        drawRect(ulx, uly, lrx, lry, 0, 0, 0, 0, false, extAlignment);
    }

    void RDP::setRectAlign(const ExtendedAlignment &extAlignment) {
        extended.global.rect = extAlignment;
    }

    void RDP::setScissorAlign(const ExtendedAlignment &extAlignment) {
        extended.global.scissor = extAlignment;
    }

    void RDP::forceUpscale2D(bool force) {
        extended.drawExtendedFlags.forceUpscale2D = force;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedFlags);
    }
    
    void RDP::forceTrueBilerp(uint8_t mode) {
        extended.drawExtendedFlags.forceTrueBilerp = mode;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedFlags);
    }

    void RDP::forceScaleLOD(bool force) {
        extended.drawExtendedFlags.forceScaleLOD = force;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedFlags);
    }

    void RDP::clearExtended() {
        extended.scissorLeftOrigin = G_EX_ORIGIN_NONE;
        extended.scissorRightOrigin = G_EX_ORIGIN_NONE;
        extended.drawExtendedFlags = {};
        extended.global.rect = ExtendedAlignment();
        extended.global.scissor = ExtendedAlignment();
    }
    
    void RDP::drawTris(uint32_t triCount, const float *pos, const float *tc, const float *col, uint8_t tile, uint8_t levels) {
        // Levels is expected to be above 0 elsewhere.
        levels += 1;

        // Check if the texture needs to be updated.
        DrawCall &drawCall = state->drawCall;
        if (!drawCall.textureOn || (drawCall.textureTile != tile) || (drawCall.textureLevels != levels)) {
            drawCall.textureOn = 1;
            drawCall.textureTile = tile;
            drawCall.textureLevels = levels;
            state->updateDrawStatusAttribute(DrawAttribute::Texture);
        }

        // Check if framebuffer pair must be changed.
        checkFramebufferPair();

        // Change projection to triangles.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        if (!fbPair.inProjection(0, Projection::Type::Triangle)) {
            state->flush();
            fbPair.changeProjection(0, Projection::Type::Triangle);
        }

        bool flushedState = state->checkDrawState();

        // We only change these once the draw state has been checked.
        drawCall.minWorldMatrix = 0;
        drawCall.maxWorldMatrix = 0;

        if (flushedState) {
            state->loadDrawState();
        }

        auto &triPosFloats = workload.drawData.triPosFloats;
        auto &triTcFloats = workload.drawData.triTcFloats;
        auto &triColorFloats = workload.drawData.triColorFloats;
        const uint32_t PosFloatsPerVertex = 4;
        const uint32_t TcFloatsPerVertex = 2;
        const uint32_t ColFloatsPerVertex = 4;
        const uint32_t PosFloatsPerTri = PosFloatsPerVertex * 3;
        const uint32_t TcFloatsPerTri = TcFloatsPerVertex * 3;
        const uint32_t ColFloatsPerTri = ColFloatsPerVertex * 3;
        triPosFloats.insert(triPosFloats.end(), pos, pos + triCount * PosFloatsPerTri);
        triTcFloats.insert(triTcFloats.end(), tc, tc + triCount * TcFloatsPerTri);
        triColorFloats.insert(triColorFloats.end(), col, col + triCount * ColFloatsPerTri);
        drawCall.triangleCount += triCount;

        // ROGUESQ_LOG_PIPELINE: log raw vertex (x,y,z,w) for cinematic-mux Triangle
        // draws. We've established these triangles reach drawInstanced but get
        // rejected by the GPU rasterizer before the pixel shader. The z-rescue
        // (clamp z to [0,w]) helped some but not all. Log the raw input
        // positions so we can spot bad w (negative, NaN, very small/large)
        // or NaN x/y that would survive z-clamp but still fail RS clip.
        {
            static const bool log_pipe = []{
                const char *e = std::getenv("ROGUESQ_LOG_PIPELINE");
                return e && *e && *e != '0';
            }();
            if (log_pipe) {
                const uint32_t lo = (uint32_t)(drawCall.colorCombiner.L);
                // 2026-05-07 — extend tri-vert capture to ALL 8 GPU-reaching
                // muxes (cinematic family + green/cyan/yellow/F5-particle).
                // Lets us spot whether non-magenta tris (suspect = explosions
                // disguised as engine effects) have stuck positions vs varied.
                const bool isCine = (lo == 0xFC11FE23u || lo == 0xFC11E623u ||
                                     lo == 0xFC119623u || lo == 0xFC127FFFu ||
                                     lo == 0xFC127E24u ||  // cyan — X-wing
                                     lo == 0xFCFFFFFFu ||  // green — engine/laser
                                     lo == 0xFC121824u ||  // yellow — untagged
                                     lo == 0x7C6E58D9u || lo == 0xBCA318B9u ||
                                     lo == 0xFC580253u || lo == 0xFCCDFCCDu);
                if (isCine) {
                    static int n = 0;
                    if (++n <= 30 || (n % 2000) == 0) {
                        // Compute summary stats over this batch's vertices.
                        const uint32_t vertCount = triCount * 3;
                        float wmin =  1e30f, wmax = -1e30f;
                        float zmin =  1e30f, zmax = -1e30f;
                        int nans = 0, wzero = 0, wneg = 0, zoutclip = 0;
                        for (uint32_t v = 0; v < vertCount; ++v) {
                            const float x = pos[v*4 + 0];
                            const float y = pos[v*4 + 1];
                            const float z = pos[v*4 + 2];
                            const float w = pos[v*4 + 3];
                            if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w)) ++nans;
                            if (w == 0.0f) ++wzero;
                            else if (w < 0.0f) ++wneg;
                            if (w > 0.0f && (z < 0.0f || z > w)) ++zoutclip;
                            wmin = std::min(wmin, w); wmax = std::max(wmax, w);
                            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
                        }
                        // Show first triangle's 3 vertices in detail.
                        const float *p = pos;
                        fprintf(stderr,
                            "[tri-vert] #%d L=0x%08X tris=%u verts=%u "
                            "stats(nan=%d w0=%d w<0=%d zoutclip=%d wmin=%.3f wmax=%.3f zmin=%.3f zmax=%.3f) "
                            "v0=(%.2f,%.2f,%.2f,%.3f) v1=(%.2f,%.2f,%.2f,%.3f) v2=(%.2f,%.2f,%.2f,%.3f)\n",
                            n, lo, triCount, vertCount,
                            nans, wzero, wneg, zoutclip, wmin, wmax, zmin, zmax,
                            p[0], p[1], p[2], p[3],
                            p[4], p[5], p[6], p[7],
                            p[8], p[9], p[10], p[11]);
                        fflush(stderr);
                    }
                }
            }
        }

        const FixedRect &scissorRect = state->rdp->scissorRectStack[scissorStackSize - 1];
        if (!scissorRect.isNull()) {
            fbPair.scissorRect.merge(scissorRect);

            FixedRect drawRect;
            for (uint32_t i = 0; i < triCount * 3; i++) {
                drawRect.ulx = std::min(drawRect.ulx, int32_t(pos[i * PosFloatsPerVertex + 0] * 4.0f));
                drawRect.uly = std::min(drawRect.uly, int32_t(pos[i * PosFloatsPerVertex + 1] * 4.0f));
                drawRect.lrx = std::max(drawRect.lrx, int32_t(ceilf(pos[i * PosFloatsPerVertex + 0]) * 4.0f));
                drawRect.lry = std::max(drawRect.lry, int32_t(ceilf(pos[i * PosFloatsPerVertex + 1]) * 4.0f));
            }

            const FixedRect intRect = scissorRect.intersection(drawRect);
            if (!intRect.isNull()) {
                fbPair.drawColorRect.merge(intRect);
                if (otherMode.zUpd()) {
                    fbPair.drawDepthRect.merge(intRect);
                }
            }
        }
    }

    void RDP::drawRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, int16_t uls, int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
        drawRect(ulx, uly, lrx, lry, uls, ult, dsdx, dtdy, flip, extended.global.rect);
    }
    
    void RDP::drawRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, int16_t uls, int16_t ult, int16_t dsdx, int16_t dtdy, bool flip, const ExtendedAlignment &extAlignment) {
        // Round down the left and top coordinates in fill mode or copy mode.
        const bool usesFillMode = (otherMode.cycleType() == G_CYC_FILL);
        const bool usesCopyMode = (otherMode.cycleType() == G_CYC_COPY);
        if (usesFillMode || usesCopyMode) {
            ulx &= ~3;
            uly &= ~3;
        }

        // Add global offsets to the coordinates.
        ulx += extAlignment.leftOffset;
        uly += extAlignment.topOffset;
        lrx += extAlignment.rightOffset;
        lry += extAlignment.bottomOffset;

        const FixedRect drawRect(movedFromOrigin(ulx, extAlignment.leftOrigin), uly, movedFromOrigin(lrx, extAlignment.rightOrigin), lry);
        if (drawRect.isEmpty()) {
            return;
        }

        // Check if framebuffer pair must be changed.
        checkFramebufferPair();

        // We always flush on any rectangle as we want to have the individual calls.
        state->flush();

        // Change projection to rectangle.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        if (!fbPair.inProjection(0, Projection::Type::Rectangle)) {
            fbPair.changeProjection(0, Projection::Type::Rectangle);
        }

        const FixedRect &scissorRect = state->rdp->scissorRectStack[scissorStackSize - 1];
        bool scissorIsNull = scissorRect.isNull();
        if (!scissorRect.isNull()) {
            fbPair.scissorRect.merge(scissorRect);

            const FixedRect intRect = scissorRect.intersection(drawRect);
            if (!intRect.isNull()) {
                fbPair.drawColorRect.merge(intRect);
                if (otherMode.zUpd()) {
                    fbPair.drawDepthRect.merge(intRect);
                }
            } else {
                // Scissor clipped this rect to nothing — the most common
                // cause of "TEXRECT submitted but invisible". Gated under
                // ROGUESQ_LOG_DPC. High-volume during cinematic if it fires.
                static const bool log_clip = []{
                    const char *a = std::getenv("ROGUESQ_LOG_ALL");
                    if (a && *a && *a != '0') return true;
                    const char *e = std::getenv("ROGUESQ_LOG_DPC");
                    return e && *e && *e != '0';
                }();
                if (log_clip) {
                    static int n = 0;
                    if (++n <= 30 || (n % 1000) == 0) {
                        fprintf(stderr, "[clip-rect] intRectNull #%d colorAddr=0x%08X scissor={%d,%d,%d,%d} draw={%d,%d,%d,%d}\n",
                            n, colorImage.address,
                            scissorRect.ulx, scissorRect.uly, scissorRect.lrx, scissorRect.lry,
                            drawRect.ulx, drawRect.uly, drawRect.lrx, drawRect.lry);
                        fflush(stderr);
                    }
                }
            }
        }
        if (scissorIsNull) {
            static const bool log_clip = []{
                const char *a = std::getenv("ROGUESQ_LOG_ALL");
                if (a && *a && *a != '0') return true;
                const char *e = std::getenv("ROGUESQ_LOG_DPC");
                return e && *e && *e != '0';
            }();
            if (log_clip) {
                static int n = 0;
                if (++n <= 30 || (n % 1000) == 0) {
                    fprintf(stderr, "[clip-rect] scissorNull #%d colorAddr=0x%08X draw={%d,%d,%d,%d}\n",
                        n, colorImage.address,
                        drawRect.ulx, drawRect.uly, drawRect.lrx, drawRect.lry);
                    fflush(stderr);
                }
            }
        }
        
        bool flushedState = state->checkDrawState();

        // We only change these once the draw state has been checked.
        DrawCall &drawCall = state->drawCall;
        drawCall.minWorldMatrix = 0;
        drawCall.maxWorldMatrix = 0;
        drawCall.rect = drawRect;
        drawCall.rectDsdx = dsdx;
        drawCall.rectDtdy = dtdy;
        drawCall.rectLeftOrigin = extAlignment.leftOrigin;
        drawCall.rectRightOrigin = extAlignment.rightOrigin;

        if (flushedState) {
            state->loadDrawState();
        }

        // Glyph-batch bind diagnostic: log loadIndex + loadCount AFTER
        // loadDrawState() has computed them. Filter for small rects (likely
        // text glyphs: width < 256 fixed-point = 64 pixels at 10.2 fixed).
        // Expected for credits text: each rect's loadCount==2 (LoadTLUT+
        // LoadBlock pair), loadIndex monotonically increasing by 2.
        {
            const int32_t rectW = drawRect.width(true, true);
            const bool glyphSized = (rectW > 0) && (rectW < 256);
            if (glyphSized) {
                static int n = 0; ++n;
                if (n <= 60 || (n % 200) == 0) {
                    if(false) fprintf(stderr, "[trace] glyph-rect #%d w=%d loadIdx=%u loadCnt=%u flushed=%d colorAddr=0x%08X\n",
                        n, rectW, drawCall.loadIndex, drawCall.loadCount,
                        (int)flushedState, colorImage.address);
                    fflush(stderr);
                }
            }
        }

        auto &triPosFloats = workload.drawData.triPosFloats;
        auto &triTcFloats = workload.drawData.triTcFloats;
        auto &triColorFloats = workload.drawData.triColorFloats;
        static const float rectPosFloats[] = {
            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 1.0f
        };

        static const float rectColorFloats[] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };

        triPosFloats.insert(triPosFloats.end(), rectPosFloats, rectPosFloats + std::size(rectPosFloats));
        triColorFloats.insert(triColorFloats.end(), rectColorFloats, rectColorFloats + std::size(rectColorFloats));

        const int32_t rectWidth = drawRect.width(true, true);
        const int32_t rectHeight = drawRect.height(true, true);
        const int32_t uvWidth = (flip ? rectHeight : rectWidth) << 2;
        const int32_t uvHeight = (flip ? rectWidth : rectHeight) << 2;
        const int32_t lrs = ((uls << 7) + dsdx * uvWidth) >> 7;
        const int32_t lrt = ((ult << 7) + dtdy * uvHeight) >> 7;
        const float vFractionOffset = (uly & 0x3) ? (dtdy >> 5) / 32.0f : 0.0f;
        float u1 = uls / 32.0f;
        float v1 = ult / 32.0f + vFractionOffset;
        float u2 = lrs / 32.0f;
        float v2 = lrt / 32.0f + vFractionOffset;
        triTcFloats.emplace_back(u1);
        triTcFloats.emplace_back(v1);
        triTcFloats.emplace_back(flip ? u1 : u2);
        triTcFloats.emplace_back(flip ? v2 : v1);
        triTcFloats.emplace_back(flip ? u2 : u1);
        triTcFloats.emplace_back(flip ? v1 : v2);
        triTcFloats.emplace_back(u2);
        triTcFloats.emplace_back(v2);
        triTcFloats.emplace_back(flip ? u1 : u2);
        triTcFloats.emplace_back(flip ? v2 : v1);
        triTcFloats.emplace_back(flip ? u2 : u1);
        triTcFloats.emplace_back(flip ? v1 : v2);
        drawCall.triangleCount += 2;

        // Update the tracked texcoords for the used tiles. Use scissor intersection to figure out 
        // what the bounds of the real sampling will be if necessary.
        const bool computeIntersection = !scissorRect.isNull() && !scissorRect.fullyInside(drawRect);
        if (computeIntersection) {
            const FixedRect intersectionRect = scissorRect.intersection(drawRect);
            if (!intersectionRect.isNull()) {
                const int32_t leftPixels = intersectionRect.left(true) - drawRect.left(true);
                const int32_t topPixels = intersectionRect.top(true) - drawRect.top(true);
                const int32_t rightPixels = intersectionRect.right(true) - drawRect.left(true);
                const int32_t bottomPixels = intersectionRect.bottom(true) - drawRect.top(true);
                const float intU1 = (((uls << 7) + dsdx * ((flip ? topPixels : leftPixels) << 2)) >> 7) / 32.0f;
                const float intV1 = (((ult << 7) + dtdy * ((flip ? leftPixels : topPixels) << 2)) >> 7) / 32.0f + vFractionOffset;
                const float intU2 = (((uls << 7) + dsdx * ((flip ? bottomPixels : rightPixels) << 2)) >> 7) / 32.0f;
                const float intV2 = (((ult << 7) + dtdy * ((flip ? rightPixels : bottomPixels) << 2)) >> 7) / 32.0f + vFractionOffset;
                updateCallTexcoords(intU1, intV1);
                updateCallTexcoords(intU2, intV2);
            }
        }
        else {
            updateCallTexcoords(u1, v1);
            updateCallTexcoords(u2, v2);
        }
    }

    void RDP::drawTexRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls, int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
        drawTexRect(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, flip, extended.global.rect);
    }
    
    void RDP::drawTexRect(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls, int16_t ult, int16_t dsdx, int16_t dtdy, bool flip, const ExtendedAlignment &extAlignment) {
        // Pipeline-stage 1 counter — drawTexRect entry. Counts every TEXRECT
        // submitted, classified by cinematic mux family. Compare to
        // [pipe-stage 2-pushDrawCall] to find drops between RDP submission
        // and InstanceDrawCall building. Gated under ROGUESQ_LOG_PIPELINE.
        {
            static const bool log_pipe = []{
                const char *e = std::getenv("ROGUESQ_LOG_PIPELINE");
                return e && *e && *e != '0';
            }();
            if (log_pipe) {
                const auto &cc = colorCombinerStack[colorCombinerStackSize - 1];
                const uint32_t lo = (uint32_t)cc.L;
                const bool isCine = (lo == 0xFC11FE23u || lo == 0xFC11E623u ||
                                     lo == 0xFC119623u || lo == 0xFC127FFFu);
                static int n_total = 0, n_cine = 0;
                ++n_total;
                if (isCine) ++n_cine;
                if ((n_total <= 5) || (n_total % 1000) == 0) {
                    fprintf(stderr,
                        "[pipe-stage 1-drawTexRect] total=%d cine=%d (lastMux=0x%08X fb=0x%08X)\n",
                        n_total, n_cine, lo, colorImage.address);
                    fflush(stderr);
                }
            }
        }
        // Geometry trace — verify TEXRECT screen coords + target fb + active
        // combiner mux. Cinematic-explosion debugging: confirms whether the
        // game is submitting RECTs with sane coords or whether they're all
        // zero-size / off-screen (rasterizer skips → no visible output).
        // Gated under ROGUESQ_LOG_DPC. High volume during cinematic.
        {
            static const bool log_geo = []{
                const char *a = std::getenv("ROGUESQ_LOG_ALL");
                if (a && *a && *a != '0') return true;
                const char *e = std::getenv("ROGUESQ_LOG_DPC");
                return e && *e && *e != '0';
            }();
            if (log_geo) {
                static int n=0;
                if (++n<=20 || (n%200)==0) {
                    const auto &cc = colorCombinerStack[colorCombinerStackSize - 1];
                    const uint64_t mux = ((uint64_t)cc.H << 32) | (uint64_t)cc.L;
                    const int32_t w = lrx - ulx;
                    const int32_t h = lry - uly;
                    fprintf(stderr,
                        "[geo-rect] #%d ul=(%d,%d) lr=(%d,%d) wh=(%d,%d) tile=%u "
                        "fb=0x%08X mux=0x%016llX cyc=%u\n",
                        n, ulx, uly, lrx, lry, w, h, tile,
                        colorImage.address,
                        (unsigned long long)mux,
                        (unsigned)otherMode.cycleType());
                    fflush(stderr);
                }
            }
        }
        // Per-rect load-binding probe. Confirms whether each glyph's TEXRECT
        // binds to its own loadOperations range (loadIndex/loadCount). For text
        // batches: expect loadCount=2 per glyph (LoadTLUT + LoadBlock) and
        // loadIndex monotonically increasing. If loadCount=0 for most rects,
        // the texture pipeline is binding stale TMEM state.
        { static int n=0; ++n; if (n<=20 || (n%500)==0) {
            const int wlc = state->ext.workloadQueue->writeCursor;
            const auto &wl = state->ext.workloadQueue->workloads[wlc];
            if(false) fprintf(stderr, "[trace] drawTexRect-bind #%d tile=%u loadOps=%zu drawRanges.loadOps.second=%zu\n",
                n, (unsigned)tile, wl.drawData.loadOperations.size(),
                wl.drawRanges.loadOperations.second);
            fflush(stderr);
        } }
#   ifdef LOG_TEXRECT_METHODS
        RT64_LOG_PRINTF("RDP::drawTexRect(ulx %d, uly %d, lrx %d, lry %d, tile %u, uls %d, ult %d, dsdx %d, dtdy %d, flip %u)", ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, flip);
#   endif

        // Check if the texture needs to be updated.
        DrawCall &drawCall = state->drawCall;
        if (!drawCall.textureOn || (drawCall.textureTile != tile) || (drawCall.textureLevels != 1)) {
            drawCall.textureOn = 1;
            drawCall.textureTile = tile;
            drawCall.textureLevels = 1;
            state->updateDrawStatusAttribute(DrawAttribute::Texture);
        }
        
        // Divide dsdx by 4 and add an extra pixel to the edges if it uses copy mode.
        const bool usesCopyMode = (otherMode.cycleType() == G_CYC_COPY);
        if (usesCopyMode) {
            dsdx >>= 2;
            lrx |= 3;
            lry |= 3;
        }

        drawRect(ulx, uly, lrx, lry, uls, ult, dsdx, dtdy, flip, extAlignment);
    }

    void RDP::updateCallTexcoords(float u, float v) {
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        for (uint32_t t = 0; t < state->drawCall.tileCount; t++) {
            DrawCallTile &callTile = workload.drawData.callTiles[state->drawCall.tileIndex + t];
            callTile.minTexcoord.x = std::min(callTile.minTexcoord.x, int(u));
            callTile.minTexcoord.y = std::min(callTile.minTexcoord.y, int(v));
            callTile.maxTexcoord.x = std::max(callTile.maxTexcoord.x, int(ceilf(u)));
            callTile.maxTexcoord.y = std::max(callTile.maxTexcoord.y, int(ceilf(v)));
        }
    }
};

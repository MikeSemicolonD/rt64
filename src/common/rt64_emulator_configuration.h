//
// RT64
//

#pragma once

#include "rt64_common.h"

namespace RT64 {
    struct EmulatorConfiguration {
        struct Dither {
            bool postBlendNoise;
            bool postBlendNoiseNegative;
        };

        struct Framebuffer {
            bool renderToRAM;
            bool copyWithGPU;
            // Copy rendered color/depth back to RDRAM. Off keeps renderToRAM's RDRAM-to-GPU change detection but skips the readback and its GPU wait.
            bool writeBack;
        };

        Dither dither;
        Framebuffer framebuffer;

        EmulatorConfiguration();
    };
};
//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct VideoInterfaceCB {
        float2 videoResolution;
        float2 textureResolution;
        float gamma;
        float viFilter;   // N64 VI-style soften radius in color-target texels; 0 = off (default)
        float2 overscan;  // right/bottom sample inset in color-target texels; keeps the composite off the target's stale edge column (0 = off)
    };
#ifdef HLSL_CPU
};
#endif
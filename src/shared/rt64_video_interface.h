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
    };
#ifdef HLSL_CPU
};
#endif
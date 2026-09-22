//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RasterParams {
        uint renderIndex;
        uint useVertexRenderIndex; // 0 = push-constant renderIndex; 1 = per-vertex RENDERINDEX input
        uint2 padding;
        float2 screenScale;
        float2 screenOffset;
    };
#ifdef HLSL_CPU
};
#endif
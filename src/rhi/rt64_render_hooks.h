//
// RT64
//

#pragma once

#include "common/rt64_plume.h"

namespace RT64 {
    using RenderHookInit = void(RenderInterface *rhi, RenderDevice *device);
    using RenderHookDraw = void(RenderCommandList *list, RenderFramebuffer *swapChainFramebuffer);
    using RenderHookDeinit = void();

    RenderHookInit *GetRenderHookInit();
    RenderHookDraw *GetRenderHookDraw();
    RenderHookDeinit *GetRenderHookDeinit();

    void SetRenderHooks(RenderHookInit *init, RenderHookDraw *draw, RenderHookDeinit *deinit);

    // Host ImGui hook: called each frame between ImGui::NewFrame and ImGui::Render
    // (inside the inspector frame), so the host can emit its own ImGui windows.
    using RenderHookImgui = void();
    RenderHookImgui *GetRenderHookImgui();
    void SetRenderHookImgui(RenderHookImgui *imgui);
};

//
// RT64 — F3DFACTOR5 diagnostics
//
// Default-off Phase-0 probes (ROGUESQ_F5_PROBE) relocated out of the GBI core.
// They log the F5 ucode's would-be interpretation of op_01/03/04 + the triangle
// commands (matrices, vertex layout, screen coords, texcoords) without changing
// the rendered output. Called from the experimental op handlers in the core,
// which gate them on f5_probe_enabled().
//
#include <cstdio>
#include <cstdint>

#include "rt64_gbi_f3dfactor5_internal.h"

namespace RT64 {
    namespace GBI_F3DFACTOR5 {

        // Probe-local MVP state (classified by m[3][3]: proj≈0, modelview≈1),
        // separate from the live render path. Only used to log would-be coords.
        static thread_local hlslpp::float4x4 s_probe_proj;
        static thread_local hlslpp::float4x4 s_probe_mv;
        static thread_local hlslpp::float4x4 s_probe_mvp;
        static thread_local bool s_probe_have_proj = false, s_probe_have_mv = false, s_probe_mvp_valid = false;

        void f5_probe_op01(uint32_t w0, uint32_t w1, const uint8_t* ram) {
            if ((w1 & 0xFF000000u) != 0x80000000u) return;
            hlslpp::float4x4 m = decode_n64_f4x4(ram, w1);
            const uint32_t flag = (w0 >> 16) & 0xFF;
            const float m33 = (float)m[3][3];
            const bool isProj = (m33 > -0.01f && m33 < 0.01f);
            const bool isMv   = (m33 > 0.99f  && m33 < 1.01f);
            if (isProj) { s_probe_proj = m; s_probe_have_proj = true; }
            else if (isMv) { s_probe_mv = m; s_probe_have_mv = true; }
            if (s_probe_have_proj && s_probe_have_mv) { s_probe_mvp = hlslpp::mul(s_probe_mv, s_probe_proj); s_probe_mvp_valid = true; }
            static int n = 0;
            if (++n <= 40) {
                std::fprintf(stderr, "[probe op01 #%d] w0=0x%08X flagByte=0x%02X bit0=%d w1=0x%08X m33=%.4f -> %s\n",
                    n, w0, flag, flag & 1, w1, m33, isProj ? "PROJ" : (isMv ? "MODELVIEW" : "?"));
                std::fflush(stderr);
            }
        }

        void f5_probe_op03(DisplayList** dl, const uint8_t* ram) {
            (void)ram;
            // op_03 command is 8 bytes at *dl; the inline 16-byte vertex follows at (*dl)+1.
            const uint32_t cw0 = (*dl)->w0, cw1 = (*dl)->w1;
            const uint8_t* vb = reinterpret_cast<const uint8_t*>(&(*dl)[1]);
            auto s16 = [&](int o){ return (int16_t)(((uint16_t)vb[o]<<8)|vb[o+1]); };
            static int n = 0;
            if (++n <= 40) {
                std::fprintf(stderr, "[probe op03 #%d] cmd w0=0x%08X w1=0x%08X | inline16:", n, cw0, cw1);
                for (int i = 0; i < 16; ++i) std::fprintf(stderr, " %02X", vb[i]);
                std::fprintf(stderr, " | xyz=(%d,%d,%d) flag=0x%04X st=(%d,%d) rgba=%02X%02X%02X%02X\n",
                    s16(0), s16(2), s16(4), (uint16_t)(((uint16_t)vb[6]<<8)|vb[7]),
                    s16(8), s16(10), vb[12], vb[13], vb[14], vb[15]);
                std::fflush(stderr);
            }
        }

        void f5_probe_op04(uint32_t w0, uint32_t w1, const uint8_t* ram) {
            if ((w1 & 0xFF000000u) != 0x80000000u) return;
            const uint32_t cnt = w0 & 0x3FF, base = (w0 >> 10) & 0x3F;
            static int n = 0;
            if (++n > 24) return;
            std::fprintf(stderr, "[probe op04 #%d] w0=0x%08X cnt=%u base=%u w1=0x%08X mvp=%d", n, w0, cnt, base, w1, s_probe_mvp_valid ? 1 : 0);
            const uint32_t lim = cnt < 3 ? cnt : 3;
            for (uint32_t i = 0; i < lim; ++i) {
                float x = (float)rd_be_s16(ram, w1 + i*8 + 0);
                float y = (float)rd_be_s16(ram, w1 + i*8 + 2);
                float z = (float)rd_be_s16(ram, w1 + i*8 + 4);
                if (s_probe_mvp_valid) {
                    float v[4] = { x, y, z, 1.0f }, o[4];
                    f5_xform_row(s_probe_mvp, v, o);
                    float cwv = o[3], iw = (cwv != 0.0f) ? 1.0f / cwv : 0.0f;
                    float sx = (o[0]*iw*0.5f + 0.5f) * 640.0f;
                    float sy = (o[1]*iw*0.5f + 0.5f) * 480.0f;
                    std::fprintf(stderr, "  v%u(%.0f,%.0f,%.0f)->scr(%.1f,%.1f) cw=%.2f", i, x, y, z, sx, sy, cwv);
                } else {
                    std::fprintf(stderr, "  v%u(%.0f,%.0f,%.0f)", i, x, y, z);
                }
            }
            std::fprintf(stderr, "\n"); std::fflush(stderr);
        }

        void f5_probe_tri(const char* tag, DisplayList** dl, State* state) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            // ucode index masks: byte offsets into the 0x670 cache, record stride 0x50.
            const uint32_t i0 = (((w1>>13)&0x7F8u)) / 0x50u;
            const uint32_t i1 = (((w1>>5)&0x7F8u)) / 0x50u;
            const uint32_t i2 = (((w1<<3)&0x7F8u)) / 0x50u;
            const uint32_t i3 = (((w1>>21)&0x7F8u)) / 0x50u;
            const bool uvb = (w0 & 0x2) != 0;
            const uint8_t* tb = reinterpret_cast<const uint8_t*>(&(*dl)[1]);
            auto s16 = [&](int o){ return (int16_t)(((uint16_t)tb[o]<<8)|tb[o+1]); };
            // Candidate next-command opcodes at +8/+16/+24 bytes — resolves the op_bf inline-block
            // SIZE (no inline → next op at +8; 8B inline → +16; 16B inline → +24). F5 op = w0>>24.
            const uint32_t op1 = (*dl)[1].w0 >> 24, op2 = (*dl)[2].w0 >> 24, op3 = (*dl)[3].w0 >> 24;
            static int n = 0;
            if (++n <= 150) {
                std::fprintf(stderr, "[probe %s #%d] w0=0x%08X w1=0x%08X uvb=%d w0&4=%d idx=%u,%u,%u,%u | inline8:",
                    tag, n, w0, w1, uvb ? 1 : 0, (w0 & 0x4) ? 1 : 0, i0, i1, i2, i3);
                for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %02X", tb[i]);
                std::fprintf(stderr, " st=(%d,%d,%d,%d) nextop@+8/+16/+24=%02X/%02X/%02X\n",
                    s16(0), s16(2), s16(4), s16(6), op1, op2, op3);
                std::fflush(stderr);
            }
        }

    } // namespace GBI_F3DFACTOR5
} // namespace RT64

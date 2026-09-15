//
// RT64
//
// HLE GBI module for Factor 5's custom RSP microcode (Rogue Squadron 64,
// Battle for Naboo, Indiana Jones and the Infernal Machine).
//
// Inherits the F3DEX dispatch table and overrides the opcodes Factor 5 reuses.
// Rebuilt 2026-09-07 from the hardware goldens (Project64 RDRAM dumps of
// cinematic frames 78/79/120, byte-identical to the recomp's chunk pool) and
// docs/f5-model-dl-spec.md after the previous working copy was lost.
//
// Stream facts this file relies on (all measured on the goldens):
//  - DL chunks are 0x108 bytes: 8-byte header (op 0x80: next/prev pointers) +
//    0x100 payload. The ucode never executes a header (0x80 dispatches to IMEM 0),
//    so a header reached by walking linearly means the sub-DL ran off its chunk.
//  - B5(0) ends a list (a full chunk has it in the last slot; the bytes after a mid-chunk
//    B5(0) are stale). ROGUESQ_OP_B5_ENDDL=0 restores the last-slot-only rule.
//  - 0x05: `05 05 .. ..` = 40-byte sprite record, `05 00 .. ..` = 8-byte command.
//  - 0xBD / 0xBE: 16-byte state commands (payload FFxxxxxx ........).
//  - 0x03: 24-byte inline lookat/light block (`03 82 ...`), viewport form is 8 bytes.
//  - 0x01: matrix load, byte1 0x03 = projection (0x8071....), 0x02 = modelview.
//  - 0x04 / 0x14: vertex batch, n = (w0>>10)&0x3F, 8-byte verts (x,y,z int16, pad).
//  - 0x02: per-vertex RGBA buffer (4 bytes per vertex, index = vertex index).
//  - 0xBF: triangle; w1 bytes = indices*5; `w0&2` = 32-byte textured form (indices*5,
//    indices*4, flags, 3 raw texcoords scaled by 03 82, pad). 0xB4: quad, same 32-byte layout.
//

#include "rt64_gbi_f3dfactor5.h"
#include "rt64_gbi_f3dfactor5_internal.h"

#include "hle/rt64_state.h"
#include "hle/rt64_rdp.h"
#include "hle/rt64_rsp.h"

#include "rt64_gbi_f3dex.h"
#include "rt64_gbi_f3d.h"
#include "rt64_gbi_rdp.h"

#include "shared/rt64_f3d_defines.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

extern "C" volatile unsigned g_most_drawn_fb = 0;
extern "C" volatile unsigned g_most_drawn_fb_width = 0;  // width of g_most_drawn_fb's color image
extern "C" volatile unsigned long long g_most_drawn_fb_ms = 0;  // steady-clock ms of its last texrect
extern "C" volatile unsigned g_f5_task_hops = 0;    // previous task's chunk transitions (diagnostic)
extern "C" volatile unsigned g_f5_task_faces = 0;   // previous task's emitted faces
extern "C" volatile int g_explosion_hold = 0;

namespace RT64 {
    namespace GBI_F3DFACTOR5 {
        // ---------------------------------------------------------------- gates
        static GBIFunction s_mw_orig = nullptr;
        static bool env_on(const char* name, bool def) {
            const char* v = std::getenv(name);
            if (!v || !v[0]) return def;
            return v[0] != '0';
        }

        // ROGUESQ_LOG_GBI=1 enables per-handler diagnostic logs.
        bool gbi_log_enabled() {
            static bool s = env_on("ROGUESQ_LOG_GBI", false);
            return s;
        }

        // ROGUESQ_HLE_FORCE_VISIBLE=1: magenta fill/combiner to test GPU output.
        bool force_visible_enabled() {
            static bool s = env_on("ROGUESQ_HLE_FORCE_VISIBLE", false);
            return s;
        }

        // ROGUESQ_F5_NATIVE=0 turns the geometry emission off (parse-only).
        static bool f5_native_active() {
            static bool s = env_on("ROGUESQ_F5_NATIVE", true);
            return s;
        }

        // ROGUESQ_LOG_FACE_UV=1: log every textured F5 face (texture addr, tile
        // fmt/size, raw + decoded UVs, object-space vertex positions) and every
        // matrix load. Diagnostic for the rotated/zoomed 3D glyph strip on the
        // name-entry screen vs the model-texture bug (transform-vs-UV).
        static bool face_uv_log_enabled() {
            static bool s = env_on("ROGUESQ_LOG_FACE_UV", false);
            return s;
        }

        // Kept for the sibling modules' declarations (the software MVP composer is gone).
        bool f5_compose_mvp(const uint8_t*, float Mout[12]) {
            for (int i = 0; i < 12; ++i) Mout[i] = 0.0f;
            return false;
        }

        // op_02's per-vertex RGBA buffer (declared in the internal header).
        thread_local uint32_t s_last_op02_colorbuf = 0, s_last_op02_colorcnt = 0;

        // Shared with rt64_gbi_f5_rdpstate.cpp (declared in the internal header).
        int s_ci4_tlut_recent = 0;
        int s_ci4_last_block_words = 0;
        uint32_t s_last_tlut_src = 0;
        int s_attrib_glyphs_frame = 0;
        int s_attrib_fills_frame = 0;
        int s_attrib_frame_idx = 0;
        int s_attrib_prev_glyphs = 0;
        thread_local int s_cart_pass = 0;

        // ------------------------------------------------------------ helpers
        static inline uint32_t f5_dl_off(State* state, const DisplayList* dl) {
            return (uint32_t)(reinterpret_cast<const uint8_t*>(dl) - state->fromRDRAM(0));
        }

        static inline bool f5_is_byte_ramp(uint32_t w) {
            const uint32_t b0 = (w >> 24) & 0xFF, b1 = (w >> 16) & 0xFF, b2 = (w >> 8) & 0xFF, b3 = w & 0xFF;
            return b1 == b0 + 1 && b2 == b0 + 2 && b3 == b0 + 3;
        }

        void op_noop(State*, DisplayList**) {}

        // Skip the 8-byte payload of a 16-byte command.
        void op_consume16(State*, DisplayList** dl) { (*dl)++; }
        // Skip the 24-byte payload of a 32-byte command.
        void op_consume32(State*, DisplayList** dl) { (*dl) += 3; }

        // ------------------------------------------------ chunk flow (from the ucode, IMEM 0x1088..0x12F8)
        // The RSP fetches one 0x108-byte chunk at a time into DMEM 0x170 and starts executing at +8
        // (the first 8 bytes are the chunk header: next-chunk pointer, prev pointer). When the cursor
        // reaches +0x108, or on op B5/0x12, it fetches the chunk named by the header's first word and
        // continues at +8. 06 pushes {chunk, cursor} and fetches w1; 07 fetches w1 without pushing;
        // B8 pops (or ends the task when the stack is empty). B5's own w1 is never read.
        static GBIFunction s_inner[UCODE_MAP_SIZE];
        static constexpr int F5_MAX_DEPTH = 64;
        static uint32_t s_task_faces = 0;
        static uint32_t s_task_budget_trip = 0;
        static void f5_ensure_viewport(State* state);
        // Above the N64's 8 MB (the recomp heap does use 0x71E000; hardware never sees this range; RT64's
        // segmented mask allows 16 MB and the host buffer is 512 MB).
        static constexpr uint32_t F5_VTX_SCRATCH = 0x00A00000u;   // 256 x 16 bytes
        static constexpr uint32_t F5_VP_SCRATCH  = 0x00A01000u;
        static constexpr uint32_t F5_FACE_SLOT   = 200;           // temp slots for per-face UVs

        struct F5Vp { int16_t vscale[4]; int16_t vtrans[4]; };   // RT64 reads halfword-swapped words
        static uint32_t s_chunk_base[F5_MAX_DEPTH];   // RDRAM offset of the chunk each level executes
        static uint64_t s_chunk_counter = ~0ull;
        static inline uint32_t f5_depth(State* state) { return (uint32_t)state->returnAddressStack.size(); }

        static inline uint32_t f5_rd32(State* state, uint32_t off) {
            const uint8_t* ram = state->RDRAM;
            return ((uint32_t)ram[off ^ 3] << 24) | ((uint32_t)ram[(off + 1) ^ 3] << 16) | ((uint32_t)ram[(off + 2) ^ 3] << 8) | ram[(off + 3) ^ 3];
        }

        // Fetch `chunk` at this level: execution continues at chunk + 8.
        static constexpr int F5_CHAIN_MAX = 256;
        static uint32_t s_chain[F5_MAX_DEPTH][F5_CHAIN_MAX];
        static uint32_t s_chain_len[F5_MAX_DEPTH];
        static uint32_t s_task_entries = 0;   // calls + branches + next-links this task (hw: a few hundred)
        // Ring of the last chunk transitions this task (printed at a budget trip; compare with the offline walk).
        struct F5Hop { uint32_t from, to, w0, w1; uint8_t depth; };
        static F5Hop s_hops[4096]; static uint32_t s_hop_n = 0;
        static void f5_enter_chunk(State* state, DisplayList** dl, uint32_t chunk) {
            ++s_task_entries;
            const uint32_t d = f5_depth(state);
            { F5Hop& h = s_hops[s_hop_n++ & 4095]; h.from = (d < (uint32_t)F5_MAX_DEPTH) ? s_chunk_base[d] : 0; h.to = chunk; h.depth = (uint8_t)d;
              h.w0 = f5_rd32(state, chunk + 8); h.w1 = f5_rd32(state, chunk + 12); }
            if (d < (uint32_t)F5_MAX_DEPTH) { s_chunk_base[d] = chunk; s_chain_len[d] = 0; }
            *dl = reinterpret_cast<DisplayList*>(state->fromRDRAM(chunk + 8)) - 1;
        }

        // Continue in the chunk named by the current chunk's header (B5 / end of chunk).
        static uint32_t s_chunk_hops = 0;   // per task; a cycle through recycled chunks would never end
        // A next-link chain revisiting one of its own chunks is a cycle through recycled memory.
        static inline bool f5_chunk_revisit(uint32_t depth, uint32_t chunk) {
            if (depth >= (uint32_t)F5_MAX_DEPTH) return false;
            uint32_t& n = s_chain_len[depth];
            for (uint32_t i = 0; i < n; ++i) if (s_chain[depth][i] == chunk) return true;
            if (n < (uint32_t)F5_CHAIN_MAX) s_chain[depth][n++] = chunk;
            return false;
        }
        static void f5_next_chunk(State* state, DisplayList** dl) {
            const uint32_t d = f5_depth(state);
            const uint32_t base = (d < (uint32_t)F5_MAX_DEPTH) ? s_chunk_base[d] : 0;
            const uint32_t next = base ? f5_rd32(state, base) : 0;
            const uint32_t target = next & 0x00FFFFFFu;
            // The allocated chunk list is doubly linked: the next chunk's prev word must point back here.
            // A stale call into a recycled chunk otherwise walks the whole free list (thousands of garbage faces).
            const bool linked = (next >> 24) == 0x80u && target != 0 && target + 0x108u <= RDRAMSize && f5_rd32(state, target + 4) == (0x80000000u | base);
            // Legitimate next-chains are short (hw goldens: <= 19 chunks per sub-list level; root uses 07 links);
            // a stale call into a recycled chunk would otherwise walk the (also doubly linked) free list.
            static const uint32_t s_chain_cap = [](){ const char* e = std::getenv("ROGUESQ_F5_CHAIN_CAP"); return (e && *e) ? (uint32_t)std::atoi(e) : 64u; }();
            const bool tooLong = d < (uint32_t)F5_MAX_DEPTH && s_chain_len[d] >= s_chain_cap;
            if (!linked || tooLong || target == base || ++s_chunk_hops > 4096u || f5_chunk_revisit(d, target)) {
                static int s_n = 0;
                if (gbi_log_enabled() && ++s_n <= 8) {
                    std::fprintf(stderr, "[gbi-f5] chunk %06X has no next (%08X); ending list\n", base, next);
                    std::fflush(stderr);
                }
                GBI_F3D::endDl(state, dl);
                return;
            }
            const uint32_t keep = (d < (uint32_t)F5_MAX_DEPTH) ? s_chain_len[d] : 0;
            f5_enter_chunk(state, dl, target);
            if (d < (uint32_t)F5_MAX_DEPTH) s_chain_len[d] = keep;   // a next-link continues the chain
        }

        // The task's first command is the root chunk header itself (RT64 starts AT data_ptr).
        void op_80_header(State* state, DisplayList** dl) {
            const uint32_t d = f5_depth(state);
            if (d < (uint32_t)F5_MAX_DEPTH) s_chunk_base[d] = f5_dl_off(state, *dl);
        }

        static bool f5_ran_off_chunk(State* state, const DisplayList* dl) {
            static bool s_on = env_on("ROGUESQ_F5_CHUNK_BOUND", true);
            if (!s_on) return false;
            const uint32_t off = f5_dl_off(state, dl);
            if (state->displayListCounter != s_chunk_counter) {   // new task: root chunk starts here
                s_chunk_counter = state->displayListCounter;
                g_f5_task_hops = s_chunk_hops; g_f5_task_faces = s_task_faces;
                // ROGUESQ_LOG_GFX_TASK: one line per finished F5 task (faces/entries/hops + whether a
                // budget cap ended it) to correlate object flicker with dropped geometry.
                { static bool s_tl = env_on("ROGUESQ_LOG_GFX_TASK", false);
                  if (s_tl) { std::fprintf(stderr, "[f5-task] done before #%llu: faces=%u entries=%u hops=%u budget_trip=%u\n",
                      (unsigned long long)s_chunk_counter, s_task_faces, s_task_entries, s_chunk_hops, s_task_budget_trip); std::fflush(stderr); } }
                s_task_budget_trip = 0;
                s_chunk_hops = 0; s_task_faces = 0; s_task_entries = 0; s_hop_n = 0;
                std::memset(s_chunk_base, 0, sizeof(s_chunk_base));
                std::memset(s_chain_len, 0, sizeof(s_chain_len));
                s_chunk_base[0] = off;
                return false;
            }
            const uint32_t d = f5_depth(state);
            return d < (uint32_t)F5_MAX_DEPTH && s_chunk_base[d] != 0 && off >= s_chunk_base[d] + 0x108u;
        }
        template <int OP>
        static void f5_bounded(State* state, DisplayList** dl) {
            if (f5_ran_off_chunk(state, *dl)) {
                f5_next_chunk(state, dl);
                return;
            }
            // Per-task budgets: a stale call into recycled chunks otherwise storms the parser with thousands of
            // garbage faces (hw goldens: < 400 chunk entries, < 700 faces per task). Ending the task drops the
            // rest of one frame instead of stalling for half a second.
            static const uint32_t s_entry_cap = [](){ const char* e = std::getenv("ROGUESQ_F5_ENTRY_CAP"); return (e && *e) ? (uint32_t)std::atoi(e) : 2048u; }();
            static const uint32_t s_face_cap = [](){ const char* e = std::getenv("ROGUESQ_F5_FACE_CAP"); return (e && *e) ? (uint32_t)std::atoi(e) : 4096u; }();
            if (s_task_entries > s_entry_cap || s_task_faces > s_face_cap) {
                s_task_budget_trip = 1;
                static int s_n = 0;
                if (gbi_log_enabled() && ++s_n <= 8) { std::fprintf(stderr, "[gbi-f5] task budget exceeded (entries %u faces %u); ending task\n", s_task_entries, s_task_faces); std::fflush(stderr); }
                // ROGUESQ_DUMP_ON_BUDGET=<path>: RDRAM snapshot at the first budget trip (walk it offline with
                // tools/validate/f5_dl_walk.py from the task's data_ptr to see where the list enters freed chunks).
                static const char* s_dump = std::getenv("ROGUESQ_DUMP_ON_BUDGET");
                static bool s_dumped = false;
                if (s_dump && *s_dump && !s_dumped) {
                    s_dumped = true;
                    const uint32_t n = s_hop_n < 4096 ? s_hop_n : 4096;
                    std::fprintf(stderr, "[gbi-f5] budget trip: last %u chunk transitions (from -> to depth: first payload words)\n", n);
                    for (uint32_t i = 0; i < n; ++i) { const F5Hop& h = s_hops[(s_hop_n - n + i) & 4095]; std::fprintf(stderr, "  %06X -> %06X d%u: %08X %08X\n", h.from, h.to, h.depth, h.w0, h.w1); }
                    if (FILE* f = std::fopen(s_dump, "wb")) {
                        std::vector<uint8_t> buf(0x800000);   // heap: a static here would be instantiated per opcode
                        for (uint32_t i = 0; i < buf.size(); ++i) buf[i] = state->RDRAM[i ^ 3];
                        std::fwrite(buf.data(), 1, buf.size(), f); std::fclose(f);
                        std::fprintf(stderr, "[gbi-f5] budget dump written: %s\n", s_dump); std::fflush(stderr);
                    }
                }
                while (!state->returnAddressStack.empty()) state->popReturnAddress();
                *dl = nullptr;
                return;
            }
            if (s_inner[OP]) {
                s_inner[OP](state, dl);
            } else {
                static int s_n = 0;
                if (gbi_log_enabled() && ++s_n <= 4) {
                    std::fprintf(stderr, "[gbi-f5] unknown op 0x%02X w0=0x%08X w1=0x%08X\n", OP, (*dl)->w0, (*dl)->w1);
                    std::fflush(stderr);
                }
            }
        }
        template <int... I>
        static void f5_install_bounded(GBI* gbi, std::integer_sequence<int, I...>) {
            ((s_inner[I] = gbi->map[I], gbi->map[I] = &f5_bounded<I>), ...);
        }

        // 0xB5 / 0x12: continue in the chunk named by the current chunk header (w1 ignored).
        void op_b5_next_chunk(State* state, DisplayList** dl) {
            f5_next_chunk(state, dl);
        }

        // 0x07: branch to chunk w1.
        void op_07_branch(State* state, DisplayList** dl) {
            const uint32_t target = state->rsp->fromSegmentedMasked((*dl)->w1);
            if (target == 0 || target + 0x108u > RDRAMSize) { GBI_F3D::endDl(state, dl); return; }
            f5_enter_chunk(state, dl, target);
        }

        // 0x06: call chunk w1 (push). Only the plain form is a call; Factor 5 reuses byte 0x06 with
        // data in the low 24 bits otherwise.
        void op_06_strict_dl(State* state, DisplayList** dl) {
            const uint32_t w0Payload = (*dl)->w0 & 0x00FEFFFF;
            if (w0Payload != 0) {
                static int s_skip = 0;
                if (gbi_log_enabled() && ++s_skip <= 8) {
                    std::fprintf(stderr, "[gbi-f5] G_DL skip w0=0x%08X w1=0x%08X\n", (*dl)->w0, (*dl)->w1);
                    std::fflush(stderr);
                }
                return;
            }
            const uint32_t target = state->rsp->fromSegmentedMasked((*dl)->w1);
            if (target == 0 || target + 0x108u > RDRAMSize) return;
            if (((*dl)->w0 >> 16) & 1) {
                f5_enter_chunk(state, dl, target);
            } else {
                state->pushReturnAddress(*dl);
                f5_enter_chunk(state, dl, target);
            }
        }

        // 0x05 `05 05 02 xx` form (ucode overlays 0xC -> 0x24): a TERRAIN TILE quad. From the record
        // (word index = 8-byte pairs after the command, halves hi/lo):
        //   w1 = (h0,h1)  word2 = (h2,h3)          per-corner heights added to y
        //   word3..word6 = RGBA colors of corners v0,v1,v2,v3
        //   word7.lo = texcoord span s (S10.5, stored as-is)   word8 = (x, y>>4)   word9 = (z, size)
        //   corners: v0=(x,y+h0,z) v1=(x+size,y+h1,z) v2=(x,y+h2,z+size) v3=(x+size,y+h3,z+size)
        //   UVs: v0 (0,s) v1 (s,s) v2 (0,0) v3 (s,0); tris (v0,v3,v2) (v0,v1,v3), current MVP.
        // The `05 05 00 xx` form DMAs a height/color grid and tessellates it (overlays 0x14/0x18): not yet.
        static void f5_tile_quad(State* state, DisplayList* rec) {
            if (!f5_native_active()) return;
            static bool s_tiles = env_on("ROGUESQ_F5_TILES", true);
            if (!s_tiles) return;
            const uint32_t w1 = rec[0].w1, w2 = rec[1].w0;
            const uint32_t col[4] = { rec[1].w1, rec[2].w0, rec[2].w1, rec[3].w0 };
            const uint32_t w7 = rec[3].w1, w8 = rec[4].w0, w9 = rec[4].w1;
            const int16_t h[4] = { (int16_t)(w1 >> 16), (int16_t)w1, (int16_t)(w2 >> 16), (int16_t)w2 };
            const int32_t x = (int16_t)(w8 >> 16), y = (int32_t)(int16_t)w8 << 4, z = (int16_t)(w9 >> 16), sz = (int16_t)w9;
            const int16_t s = (int16_t)(w7 & 0xFFFF);   // stored into the vertex as-is by overlay 0x24 (S10.5)
            f5_ensure_viewport(state);
            RSP::Vertex* tmp = reinterpret_cast<RSP::Vertex*>(state->fromRDRAM(F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)));
            const int32_t px[4] = { x, x + sz, x, x + sz }, pz[4] = { z, z, z + sz, z + sz };
            const int16_t us[4] = { 0, s, 0, s }, vt[4] = { s, s, 0, 0 };
            for (int k = 0; k < 4; ++k) {
                tmp[k].x = (int16_t)px[k]; tmp[k].y = (int16_t)(y + h[k]); tmp[k].z = (int16_t)pz[k]; tmp[k].flag = 0;
                tmp[k].s = us[k]; tmp[k].t = vt[k];
                tmp[k].color.r = (uint8_t)(col[k] >> 24); tmp[k].color.g = (uint8_t)(col[k] >> 16);
                tmp[k].color.b = (uint8_t)(col[k] >> 8);  tmp[k].color.a = (uint8_t)col[k];
            }
            state->rsp->setVertex(0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)), 4, F5_FACE_SLOT);
            state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 3, F5_FACE_SLOT + 2);
            state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 1, F5_FACE_SLOT + 3);
            s_task_faces += 2;
        }

        // 0x05 `05 05 00 xx` form: an N*N SIGNED-height terrain tile (HMP tile height_values, N=5 in practice).
        // rec[1].w0 -> 25 s8 heights, rec[1].w1 -> 25 RGBA8888 vertex colors (+ per-tile LOD in low bytes),
        // w7.lo = texcoord span, w8/w9 = tile x,y,z,size (same encoding as the flat tile). Heights are the
        // flat s16 heights >>4, so worldY = base + h*16 (ROGUESQ_F5_TERRAIN_HSCALE multiplies). The 5x5 is
        // bilinear-subdivided to a finer mesh (ROGUESQ_F5_TERRAIN_SUB, default 2) for smooth slopes.
        static void f5_tile_grid(State* state, DisplayList* rec) {
            if (!f5_native_active()) return;
            static bool s_grid = env_on("ROGUESQ_F5_TERRAIN", true);
            if (!s_grid) return;
            static float s_hmul = 1.0f; static int s_hmul_set = 0;   // extra user tuning multiplier (rerogue-derived coeff = 1.0)
            if (!s_hmul_set) { s_hmul_set = 1; const char* v = std::getenv("ROGUESQ_F5_TERRAIN_HSCALE"); if (v && *v) s_hmul = (float)std::atof(v); }
            const uint32_t hptr = rec[1].w0 & 0x00FFFFFFu;   // N*N s8 heights
            const uint32_t cptr = rec[1].w1 & 0x00FFFFFFu;   // N*N RGBA8888 vertex colors
            // Parser overlay 0xC: byte1 = N samples per row, byte3 = height stride shift, byte2>>4 = color
            // LOD shift. Output = M*M samples, M = ((N-1)>>shift)+1, spaced `size` apart (overlay 0x10),
            // so the tile spans (M-1)*size: far tiles N=5 shift=1 size=256, near N=5 shift=0 size=128.
            const int gridN = (int)((rec[0].w0 >> 16) & 0xFF) > 1 ? (int)((rec[0].w0 >> 16) & 0xFF) : 5;
            const int shift = (int)(rec[0].w0 & 0xFF) & 7;
            const int M = ((gridN - 1) >> shift) + 1;
            const uint32_t w7 = rec[3].w1, w8 = rec[4].w0, w9 = rec[4].w1;
            const int32_t x = (int16_t)(w8 >> 16), y = (int32_t)(int16_t)w8 << 4, z = (int16_t)(w9 >> 16), sz = (int16_t)w9;
            // Overlay 0x10: s += w7.lo per sample along x, t = w7.lo*(M-1) - w7.lo*j along z (S10.5, no scale).
            const int16_t s = (int16_t)(w7 & 0xFFFF);
            const float span = (float)((M - 1) * sz);          // tile extent in world units
            const float step5 = span / 4.0f;                    // world units per 5x5-space cell
            const float uvcell = (float)s * (float)(M - 1) / 4.0f;   // S10.5 per 5x5-space cell
            const uint8_t* ram = state->RDRAM;
            // Grid heights are the flat-tile s16 heights compressed to s8 (>>4): flat corner heights run
            // 544..2032, grid s8 run 31..110, ratio ~16. So grid worldY = h<<4 to seat against flat tiles
            // (using a smaller scale drops grid tiles below the flat plateaus -> stepped layers). Env mult tunes.
            const float hcoeff = 16.0f * s_hmul;
            f5_ensure_viewport(state);
            int8_t H[25];
            // Resample the N*N height grid onto the 5x5 the mesh below is built from (nearest for N != 5).
            for (int r = 0; r < 5; ++r) for (int c = 0; c < 5; ++c) {
                const int gr = (gridN == 5) ? r : (int)(r * (gridN - 1) / 4.0f + 0.5f), gc = (gridN == 5) ? c : (int)(c * (gridN - 1) / 4.0f + 0.5f);
                H[r * 5 + c] = (int8_t)ram[(hptr + (uint32_t)(gr * gridN + gc)) ^ 3];
            }
            // ROGUESQ_LOG_GFX_TASK: trace near-LOD (shift 0) and blended tiles to catch transition garbage.
            { static bool s_lg = env_on("ROGUESQ_LOG_GFX_TASK", false); static int s_n = 0;
              const uint32_t w5 = rec[2].w1 & 0xFFFF, w6 = rec[3].w0 & 0xFFFF;
              bool blank = true; for (int k = 0; k < 25 && blank; ++k) blank = (H[k] == 0);
              if (s_lg && (shift == 0 || w5 || w6 || blank) && ++s_n <= 600) {
                  int hmin = 127, hmax = -128; for (int k = 0; k < 25; ++k) { if (H[k] < hmin) hmin = H[k]; if (H[k] > hmax) hmax = H[k]; }
                  std::fprintf(stderr, "[f5-tile] task=%llu x=%d y=%d z=%d size=%d N=%d shift=%d w5=%04X w6=%04X hptr=%06X h=[%d,%d] blank=%d" "\n",
                      (unsigned long long)state->displayListCounter, x, y, z, sz, gridN, shift, w5, w6, hptr, hmin, hmax, (int)blank); std::fflush(stderr); } }
            // Stitch to flat neighbors: a grid edge facing a coarse (flat-rendered) neighbor must be a
            // straight line between its corners, else its subdivided intermediate verts T-junction with the
            // flat quad. Detect flat vs grid neighbors from the level tile grid (D_80136DC0): grid cells have
            // tile-index top bits 110 (raw & 0xE000 == 0xC000), flat/coarse cells don't. Verified offline.
            bool stL = false, stR = false, stU = false, stD = false;
            {
                static bool s_stitch = env_on("ROGUESQ_F5_TERRAIN_STITCH", true);
                if (s_stitch) {
                    auto rd32 = [&](uint32_t a){ return ((uint32_t)ram[a^3]<<24)|((uint32_t)ram[(a+1)^3]<<16)|((uint32_t)ram[(a+2)^3]<<8)|(uint32_t)ram[(a+3)^3]; };
                    auto rd16 = [&](uint32_t a){ return (uint32_t)((ram[a^3]<<8)|ram[(a+1)^3]); };
                    const uint32_t idxArr = rd32(0x136DC0) & 0x00FFFFFFu, tileData = rd32(0x136DC4) & 0x00FFFFFFu;
                    const uint32_t gw = rd16(0x136DF8), gh = rd16(0x136DFA);   // hdr +0x38 width, +0x3A height
                    const int csz = (int)span;                                 // tile world spacing
                    if (idxArr > 0x1000 && idxArr < RDRAMSize && tileData > 0x1000 && tileData < RDRAMSize &&
                        gw > 0 && gw <= 256 && gh > 0 && gh <= 256 && csz > 0) {
                        static uint32_t s_idxArr = 0; static int s_ox = 0, s_oz = 0; static bool s_ook = false;
                        if (idxArr != s_idxArr) s_ook = false;                 // new level: re-derive origin
                        if (!s_ook) {                                         // derive origin from this tile's cell
                            const uint32_t ti = (hptr - 5u - tileData) / 0x1Eu;
                            for (uint32_t r2 = 0; r2 < gh && !s_ook; ++r2) for (uint32_t c2 = 0; c2 < gw; ++c2)
                                if ((rd16(idxArr + (r2 * gw + c2) * 2) & 0x1FFF) == ti) {
                                    s_ox = x - (int)c2 * csz; s_oz = z - (int)r2 * csz; s_idxArr = idxArr; s_ook = true; break; }
                        }
                        if (s_ook) {
                            const int col = (x - s_ox) / csz, row = (z - s_oz) / csz;
                            auto flatNb = [&](int c, int r) -> bool {         // true = coarse neighbor -> straighten edge
                                if (c < 0 || r < 0 || c >= (int)gw || r >= (int)gh) return false;   // off-map: no crack
                                return (rd16(idxArr + ((uint32_t)r * gw + (uint32_t)c) * 2) & 0xE000u) != 0xC000u; };
                            stL = flatNb(col - 1, row); stR = flatNb(col + 1, row);
                            stU = flatNb(col, row - 1); stD = flatNb(col, row + 1);
                        }
                    }
                }
            }
            // Subdivision: the ucode (overlay 0x14) subdivides the 5x5 into a finer interpolated mesh for
            // smooth slopes; raw 5x5 quads look faceted. Bilinear-interpolate to (4*S+1)^2 verts. S from env
            // (per-tile LOD lives in w1 low bytes; uniform S avoids inter-tile cracks). Grid tiles span (M-1)*size
            // (step5 per 5x5 cell); UV spans the tile over the 4 cells; colors are read at the hardware stride
            // (real values at even samples, odd are checkerboard filler) so sample the nearest even cell.
            static int s_sub = -1;
            if (s_sub < 0) { const char* v = std::getenv("ROGUESQ_F5_TERRAIN_SUB"); s_sub = (v && *v) ? std::atoi(v) : 2; if (s_sub < 1) s_sub = 1; if (s_sub > 4) s_sub = 4; }
            const int N = 4 * s_sub;   // cells per axis; N+1 verts per axis; 2*(N+1) <= 34 slots for S<=4
            RSP::Vertex* tmp = reinterpret_cast<RSP::Vertex*>(state->fromRDRAM(F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)));
            auto emit_vertex = [&](RSP::Vertex& v, float fx, float fy) {
                // On an edge facing a flat neighbor, follow the straight line between the tile's edge corners
                // (matches the flat quad's straight edge); otherwise bilinear-interpolate the 5x5 samples.
                float h;
                if      (fx <= 0.0f && stL) h = (float)H[0] * (1.0f - fy/4.0f) + (float)H[20] * (fy/4.0f);   // left edge (col 0)
                else if (fx >= 4.0f && stR) h = (float)H[4] * (1.0f - fy/4.0f) + (float)H[24] * (fy/4.0f);   // right edge (col 4)
                else if (fy <= 0.0f && stU) h = (float)H[0] * (1.0f - fx/4.0f) + (float)H[4]  * (fx/4.0f);   // top edge (row 0)
                else if (fy >= 4.0f && stD) h = (float)H[20]* (1.0f - fx/4.0f) + (float)H[24] * (fx/4.0f);   // bottom edge (row 4)
                else {
                    int x0 = (int)fx, y0 = (int)fy; if (x0 > 3) x0 = 3; if (y0 > 3) y0 = 3;
                    const float tx = fx - x0, ty = fy - y0;
                    h = ((float)H[y0*5+x0]*(1-tx) + (float)H[y0*5+x0+1]*tx) * (1-ty)
                      + ((float)H[(y0+1)*5+x0]*(1-tx) + (float)H[(y0+1)*5+x0+1]*tx) * ty;
                }
                v.x = (int16_t)(x + (int32_t)(fx * step5));
                v.y = (int16_t)(y + (int32_t)(h * hcoeff));
                v.z = (int16_t)(z + (int32_t)(fy * step5));
                v.flag = 0; v.s = (int16_t)(uvcell * fx); v.t = (int16_t)(uvcell * (4.0f - fy));
                // Colors: hardware reads every (1<<shift)-th sample of the N*N color grid.
                const int cstep = 1 << shift;
                int cx = (int)((fx * (gridN - 1) / 4.0f) / cstep + 0.5f) * cstep, cy = (int)((fy * (gridN - 1) / 4.0f) / cstep + 0.5f) * cstep;
                if (cx > gridN - 1) cx = gridN - 1; if (cy > gridN - 1) cy = gridN - 1;
                const uint32_t cb = cptr + (uint32_t)(cy * gridN + cx) * 4;
                v.color.r = ram[(cb + 0) ^ 3]; v.color.g = ram[(cb + 1) ^ 3];
                v.color.b = ram[(cb + 2) ^ 3]; v.color.a = ram[(cb + 3) ^ 3];
            };
            const uint32_t base = 0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex));
            for (int rr = 0; rr < N; ++rr) {   // one 2-row strip at a time
                for (int cc = 0; cc <= N; ++cc) {
                    emit_vertex(tmp[cc],           (float)cc / s_sub, (float)rr / s_sub);
                    emit_vertex(tmp[(N + 1) + cc], (float)cc / s_sub, (float)(rr + 1) / s_sub);
                }
                state->rsp->setVertex(base, 2 * (N + 1), F5_FACE_SLOT);
                for (int cc = 0; cc < N; ++cc) {
                    const int v0 = F5_FACE_SLOT + cc, v2 = F5_FACE_SLOT + (N + 1) + cc;
                    state->rsp->drawIndexedTri(v0, v2 + 1, v2);      // (v0,v3,v2)
                    state->rsp->drawIndexedTri(v0, v0 + 1, v2 + 1);  // (v0,v1,v3)
                }
            }
            s_task_faces += 2;   // count the DL record (runaway guard = ~faces/task), NOT the subdivided tris
        }

        void op_05_record(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0;
            // ROGUESQ_LOG_GFX_TASK: report op-05 headers whose byte1 (samples per row) is not 5, the only
            // record shape the parser below handles; anything else is walked as an 8-byte command.
            { static bool s_lg = env_on("ROGUESQ_LOG_GFX_TASK", false); static int s_n = 0;
              if (s_lg && ((w0 >> 16) & 0xFFu) != 0x05u && ++s_n <= 40) {
                  std::fprintf(stderr, "[f5-op05] unusual header %08X %08X next %08X %08X\n", w0, (*dl)->w1, (*dl)[1].w0, (*dl)[1].w1); std::fflush(stderr); } }
            if (((w0 >> 16) & 0xFFu) != 0x05u) return;      // plain 8-byte command
            if ((w0 >> 8) & 0x2u) f5_tile_quad(state, *dl);   // 05 05 02 = flat tile
            else                  f5_tile_grid(state, *dl);   // 05 05 00 = heightfield grid
            (*dl) += 4;                                       // 40-byte record
        }

        // 0x03: 24 bytes: byte 1 selects a DMEM slot, the next 16 bytes are stored there inline.
        // 0x80 = viewport (vscale x,y,z,pad, vtrans x,y,z,pad in 2-bit fixed).
        // 0x82 = texcoord scale at DMEM 0x140: 4 hi halfwords (s,t,s,t) then 4 lo halfwords, 16.16.
        // The tri path multiplies each raw per-face UV by it (vmudn lo / vmadh hi) to get S10.5;
        // the game sends (W-1)/128 per material, so raw UVs are 4.12 normalized, not 8.8 texels.
        static thread_local int32_t s_tc_scale[2] = { 0, 0 };
        static void f5_set_viewport(State* state, const int16_t* vs, const int16_t* vt);
        void op_03_f5(State* state, DisplayList** dl) {
            const uint32_t sub = ((*dl)->w0 >> 16) & 0xFF;
            if (sub == 0x80) {
                const uint32_t a = (*dl)[1].w0, b = (*dl)[1].w1, c = (*dl)[2].w0, d = (*dl)[2].w1;
                const int16_t vs[4] = { (int16_t)(a >> 16), (int16_t)a, (int16_t)(b >> 16), (int16_t)b };
                const int16_t vt[4] = { (int16_t)(c >> 16), (int16_t)c, (int16_t)(d >> 16), (int16_t)d };
                f5_set_viewport(state, vs, vt);
            } else if (sub == 0x82) {
                const uint32_t hi = (*dl)[1].w0, lo = (*dl)[2].w0;
                s_tc_scale[0] = (int32_t)((hi & 0xFFFF0000u) | (lo >> 16));
                s_tc_scale[1] = (int32_t)((hi << 16) | (lo & 0xFFFFu));
            }
            (*dl) += 2;
        }

        // ------------------------------------------------ geometry state
        // Scratch RDRAM (zero on hardware in every golden): converted vertices + a viewport.

        static thread_local uint32_t s_cache_count = 0;
        static thread_local uint32_t s_vp_key = 0;

        // Factor 5's stream never sends G_MW_CLIP; stock ucode sets clip ratio 2 (guard band) in reset().
        // Without it the F5 path keeps the RSP default (1 = no guard band), so large/close geometry and
        // near terrain get clipped at the exact viewport edges. Match stock: apply ratio 2 (env-tunable).
        static void f5_apply_clip_ratio(State* state) {
            static int s_cr = -1;
            if (s_cr < 0) { const char* v = std::getenv("ROGUESQ_F5_CLIPRATIO"); s_cr = (v && *v) ? std::atoi(v) : 2; if (s_cr < 1) s_cr = 1; if (s_cr > 15) s_cr = 15; }
            state->rsp->setClipRatioAll((int16_t)s_cr);
        }

        // Viewport from the stream's inline `03 80` block (2-bit fixed, N64 Vp layout).
        static void f5_set_viewport(State* state, const int16_t* vs, const int16_t* vt) {
            F5Vp* vp = reinterpret_cast<F5Vp*>(state->fromRDRAM(F5_VP_SCRATCH));
            // Factor 5's NDC is y-down (the ucode adds ndc*vscale with no negation; hw frame 300 puts the
            // X-wing at ndc y +0.8 = bottom of the screen). RT64 negates y for F3D, so hand it -vscale.y.
            static bool s_flip = env_on("ROGUESQ_F5_FLIP_Y", true);
            vp->vscale[1] = vs[0]; vp->vscale[0] = (int16_t)(s_flip ? -vs[1] : vs[1]); vp->vscale[3] = vs[2]; vp->vscale[2] = vs[3];
            vp->vtrans[1] = vt[0]; vp->vtrans[0] = vt[1]; vp->vtrans[3] = vt[2]; vp->vtrans[2] = vt[3];
            state->rsp->setViewport(0x80000000u | F5_VP_SCRATCH);
            f5_apply_clip_ratio(state);
            s_vp_key = 0xFFFFFFFFu;   // stream-provided: no synthesis until the next task
        }

        // Factor 5's stream never sends G_VIEWPORT (its ucode maps to the screen itself), so derive
        // one from the current scissor rectangle whenever it changes.
        static void f5_ensure_viewport(State* state) {
            const FixedRect& r = state->rdp->scissorRectStack[state->rdp->scissorStackSize - 1];
            int w = (r.lrx - r.ulx) / 4, h = (r.lry - r.uly) / 4;
            if (w <= 0 || w > 1024 || h <= 0 || h > 1024) { w = 320; h = 240; }
            const uint32_t key = ((uint32_t)w << 16) | (uint32_t)h;
            if (key == s_vp_key || s_vp_key == 0xFFFFFFFFu) return;
            s_vp_key = key;
            F5Vp* vp = reinterpret_cast<F5Vp*>(state->fromRDRAM(F5_VP_SCRATCH));
            const int16_t xs = (int16_t)(w * 2), ys = (int16_t)(h * 2);   // half-size in 2-bit fixed
            static bool s_flip2 = env_on("ROGUESQ_F5_FLIP_Y", true);
            vp->vscale[1] = xs; vp->vscale[0] = (int16_t)(s_flip2 ? -ys : ys); vp->vscale[3] = 511; vp->vscale[2] = 0;
            vp->vtrans[1] = (int16_t)(xs + r.ulx / 2); vp->vtrans[0] = (int16_t)(ys + r.uly / 2); vp->vtrans[3] = 511; vp->vtrans[2] = 0;
            state->rsp->setViewport(0x80000000u | F5_VP_SCRATCH);
            f5_apply_clip_ratio(state);
        }

        // 0x01: matrix load. byte1 0x03 = projection, 0x02 = modelview; a matrix whose bottom-right
        // element is 0 with a non-zero [2][3] is a projection whatever the byte says.
        void op_01_matrix(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            const uint32_t addr = state->rsp->fromSegmentedMasked(w1);
            if (addr + 64 > RDRAMSize) return;
            const uint8_t* ram = state->RDRAM;
            const int m33 = rd_be_s16(ram, w1 + 2 * 15), m23 = rd_be_s16(ram, w1 + 2 * 11);
            const bool proj = (((w0 >> 16) & 0xFF) == 0x03) || (m33 == 0 && m23 != 0);
            if (!f5_native_active()) return;
            f5_ensure_viewport(state);
            state->rsp->matrix(w1, proj ? 0x03 : 0x02);   // F3D constants: PROJECTION=1, LOAD=2

            // ROGUESQ_LOG_FACE_UV: compose the full fixed-point matrix (int part + frac/65536)
            // and print its top-left 3x3 + translation. A ~90-deg rotation in the upper-left
            // 2x2 next to a rotated glyph face pins the defect on the transform, not the UVs.
            if (face_uv_log_enabled()) {
                static uint64_t s_mtask = ~0ull; static int s_m = 0;
                if (state->displayListCounter != s_mtask) { s_mtask = state->displayListCounter; s_m = 0; }
                if (++s_m <= 16) {
                    float M[16];
                    for (int i = 0; i < 16; ++i)
                        M[i] = (float)rd_be_s16(ram, w1 + 2 * i) + (float)rd_be_u16(ram, w1 + 32 + 2 * i) / 65536.0f;
                    std::fprintf(stderr,
                        "[face-mtx #%d] %s addr=%06X r0[% .3f % .3f % .3f] r1[% .3f % .3f % .3f] r2[% .3f % .3f % .3f] tr[% .1f % .1f % .1f]\n",
                        s_m, proj ? "PROJ" : "MDLV", w1 & 0x00FFFFFFu,
                        M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10], M[12], M[13], M[14]);
                    std::fflush(stderr);
                }
            }
            // F5 frame interpolation is RT64's built-in AUTO geometric matcher (enable via ROGUESQ_RT_INTERP
            // -> UserConfiguration.refreshRate). Explicit per-object matrixId stamping was tried extensively
            // (entity-ptr, submit-order queue, mtx_ptr map, address-derived) and all mis-pair -> geometry
            // warp/flicker: F5 gives no stable deliverable per-object id. AUTO alone is best. See the plan.
        }

        // 0x02: per-vertex RGBA staging buffer for the next vertex batch.
        void op_02_colors(State*, DisplayList** dl) {
            s_last_op02_colorbuf = (*dl)->w1;
            s_last_op02_colorcnt = ((*dl)->w0 & 0xFFFu) + 1u;
        }

        // 0x04 / 0x14: vertex batch. Converts the 8-byte vertices + staged colors into N64 Vtx in
        // scratch RDRAM and loads them into RT64's cache at slot 0.
        void op_04_vertex(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            const uint32_t n = (w0 >> 10) & 0x3F;
            const uint32_t src = state->rsp->fromSegmentedMasked(w1);
            if (n == 0 || src + n * 8 > RDRAMSize) return;
            if (!f5_native_active()) return;
            f5_ensure_viewport(state);
            const uint8_t* ram = state->RDRAM;
            const uint32_t cbuf = s_last_op02_colorbuf;
            const bool haveColors = cbuf >= 0x80000000u && ((cbuf & 0x00FFFFFFu) + n * 4) <= RDRAMSize;
            RSP::Vertex* out = reinterpret_cast<RSP::Vertex*>(state->fromRDRAM(F5_VTX_SCRATCH));
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t va = w1 + i * 8;
                out[i].x = (int16_t)rd_be_s16(ram, va);
                out[i].y = (int16_t)rd_be_s16(ram, va + 2);
                out[i].z = (int16_t)rd_be_s16(ram, va + 4);
                out[i].flag = 0;
                out[i].s = 0;
                out[i].t = 0;
                const uint32_t c = haveColors ? rd_be_u32(ram, cbuf + i * 4) : 0xFFFFFFFFu;
                out[i].color.r = (uint8_t)(c >> 24);
                out[i].color.g = (uint8_t)(c >> 16);
                out[i].color.b = (uint8_t)(c >> 8);
                out[i].color.a = (uint8_t)c;
            }
            state->rsp->setVertex(0x80000000u | F5_VTX_SCRATCH, n, 0);
            s_cache_count = n;
        }

        // raw UV * 16.16 scale -> S10.5, as the ucode's vmudn/vmadh pair (result clamped to int16).
        static inline int16_t f5_tc_apply(int16_t raw, int32_t scale) {
            const int64_t v = ((int64_t)raw * (int64_t)scale) >> 16;
            return (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
        }

        // Emit one face from cache indices. Texcoords are per-face, so the referenced vertices are
        // re-emitted into temp slots with their UVs scaled by the current 03 82 texcoord scale.
        static void f5_emit_face(State* state, const uint32_t* idx, int n, const uint32_t* st, uint32_t colorWord) {
            if (!f5_native_active()) return;
            RSP::Vertex* tmp = reinterpret_cast<RSP::Vertex*>(state->fromRDRAM(F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)));
            const uint32_t cbuf = s_last_op02_colorbuf;
            const bool haveColors = cbuf >= 0x80000000u && ((cbuf & 0x00FFFFFFu) + 0x100u) <= RDRAMSize;
            // word 2 = per-vertex byte offsets into the op_02 color buffer (v0..v3 in bytes 1,2,3,0)
            const uint32_t cofs[4] = { (colorWord >> 16) & 0xFF, (colorWord >> 8) & 0xFF, colorWord & 0xFF, colorWord >> 24 };
            for (int k = 0; k < n; ++k) {
                if (idx[k] >= F5_FACE_SLOT || idx[k] >= s_cache_count) return;   // garbage / stale index
                tmp[k] = state->rsp->vertices[idx[k]];
                if (haveColors) {
                    const uint32_t c = rd_be_u32(state->RDRAM, cbuf + cofs[k]);
                    tmp[k].color.r = (uint8_t)(c >> 24); tmp[k].color.g = (uint8_t)(c >> 16);
                    tmp[k].color.b = (uint8_t)(c >> 8);  tmp[k].color.a = (uint8_t)c;
                }
                if (st) {
                    // ROGUESQ_F5_TC_SCALE=0: legacy raw/8 (treats raw as 8.8 texels; ignores 03 82).
                    static bool s_scale = env_on("ROGUESQ_F5_TC_SCALE", true);
                    const int16_t rs = (int16_t)(st[k] >> 16), rt = (int16_t)(st[k] & 0xFFFF);
                    if (s_scale) {
                        tmp[k].s = f5_tc_apply(rs, s_tc_scale[0]);
                        tmp[k].t = f5_tc_apply(rt, s_tc_scale[1]);
                    } else {
                        tmp[k].s = (int16_t)(rs / 8);
                        tmp[k].t = (int16_t)(rt / 8);
                    }
                }
            }
            // ROGUESQ_LOG_FACE_UV: per textured face, the source texture + tile format + raw/decoded
            // UVs + object-space vertex positions. Group by tex= to tell the glyph strip from the
            // X-wing preview; compare UV order vs position order to catch a transposed-UV (rotation).
            if (st && face_uv_log_enabled()) {
                static uint64_t s_futask = ~0ull; static int s_fu = 0;
                if (state->displayListCounter != s_futask) { s_futask = state->displayListCounter; s_fu = 0; }
                if (++s_fu <= 24) {
                    const uint32_t texaddr = state->rdp->texture.address & 0x00FFFFFFu;
                    const LoadTile& T = state->rdp->tiles[0];
                    std::fprintf(stderr, "[face-uv #%d] n=%d tex=%06X fmt=%u siz=%u line=%u uls=%u lrs=%u |",
                        s_fu, n, texaddr, T.fmt, T.siz, T.line, T.uls, T.lrs);
                    for (int k = 0; k < n; ++k)
                        std::fprintf(stderr, " v%d raw=%08X s=%d t=%d pos=(%d,%d,%d)",
                            k, st[k], (int)tmp[k].s, (int)tmp[k].t, (int)tmp[k].x, (int)tmp[k].y, (int)tmp[k].z);
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
            }
            // Copy cycle type + triangles is undefined on hardware (RT64 asserts): a stale-walk symptom, skip the face.
            if (state->rdp->otherMode.cycleType() == G_CYC_COPY) { ++s_task_faces; return; }
            // DIAG (ROGUESQ_LOG_GBI): the filter/cycle a MODEL face actually draws with. Point vs bilerp
            // decides whether models are blocky (point) or smooth. Bounded to the first few faces.
            { static int s_ff = 0; if (gbi_log_enabled() && ++s_ff <= 8) {
                std::fprintf(stderr, "[f5-face] n=%d textFilt=%u cycle=%u otherH=0x%08X otherL=0x%08X\n",
                    n, state->rdp->otherMode.textFilt(), state->rdp->otherMode.cycleType(),
                    state->rdp->otherMode.H, state->rdp->otherMode.L);
                std::fflush(stderr); } }
            state->rsp->setVertex(0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)), n, F5_FACE_SLOT);
            state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 1, F5_FACE_SLOT + 2);
            if (n == 4) state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 2, F5_FACE_SLOT + 3);
            ++s_task_faces;
        }

        // 0xBF: triangle. 16 bytes (cmd, indices*4 + flags); `w0&2` adds 8 bytes = per-face UVs
        // (words 3..5 of the 24-byte form).
        void op_bf_tri(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            const bool textured = (w0 & 0x2) != 0;
            const uint32_t idx[3] = { ((w1 >> 16) & 0xFF) / 5u, ((w1 >> 8) & 0xFF) / 5u, (w1 & 0xFF) / 5u };
            g_op_bf_count = g_op_bf_count + 1;
            if (textured) {
                // 32 bytes like 0xB4: [cmd][idx*5][idx*4][flags][st0][st1][st2][pad]
                const uint32_t st[3] = { (*dl)[2].w0, (*dl)[2].w1, (*dl)[3].w0 };
                f5_emit_face(state, idx, 3, st, (*dl)[1].w0);
                (*dl) += 3;
            } else {
                f5_emit_face(state, idx, 3, nullptr, (*dl)[1].w0);
                (*dl) += 1;
            }
        }

        // 0xB4: oriented quad, 32 bytes: indices*5 (w1), indices*4, flags, 4 UVs. Chunk-tail filler
        // (byte ramps) means the walk left the list: return.
        void op_b4_quad(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            static bool s_guard = !env_on("ROGUESQ_F5_NOFILLERGUARD", false);
            if (s_guard && (f5_is_byte_ramp(w0) || f5_is_byte_ramp(w1))) {
                *dl = state->popReturnAddress();
                return;
            }
            // Ucode order: bytes 1,2,3 then byte 0; colors/UVs follow it, tris (0,1,2) (0,2,3).
            const uint32_t idx[4] = { ((w1 >> 16) & 0xFF) / 5u, ((w1 >> 8) & 0xFF) / 5u, (w1 & 0xFF) / 5u, (w1 >> 24) / 5u };
            // Textured (w0&2) = 32 bytes with a 4-UV block; untextured = 16 bytes, no UVs. Mirrors
            // op_13_quad and the hardware stride: an unconditional 32 reads the next command as bogus UVs
            // AND over-advances 16 bytes, running the walk away on any DL with untextured 0xB4 (proven vs
            // cine_frame300 with tools/validate/f5_dl_walk.py --b4). ROGUESQ_F5_B4_LEN16 forces the 16B path.
            static bool s_len16 = env_on("ROGUESQ_F5_B4_LEN16", false);
            if (!s_len16 && (w0 & 0x2)) {
                const uint32_t st[4] = { (*dl)[2].w0, (*dl)[2].w1, (*dl)[3].w0, (*dl)[3].w1 };
                f5_emit_face(state, idx, 4, st, (*dl)[1].w0);
                (*dl) += 3;
            } else {
                f5_emit_face(state, idx, 4, nullptr, (*dl)[1].w0);
                (*dl) += 1;
            }
        }

        // 0x13: quad variant with the 0xBF framing (4 indices in w1; `w0&2` = UV block).
        void op_13_quad(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            // Ucode order: bytes 1,2,3 then byte 0; colors/UVs follow it, tris (0,1,2) (0,2,3).
            const uint32_t idx[4] = { ((w1 >> 16) & 0xFF) / 5u, ((w1 >> 8) & 0xFF) / 5u, (w1 & 0xFF) / 5u, (w1 >> 24) / 5u };
            if (w0 & 0x2) {
                const uint32_t st[4] = { (*dl)[2].w0, (*dl)[2].w1, (*dl)[3].w0, (*dl)[3].w1 };
                f5_emit_face(state, idx, 4, st, (*dl)[1].w0);
                (*dl) += 3;
            } else {
                f5_emit_face(state, idx, 4, nullptr, (*dl)[1].w0);
                (*dl) += 1;
            }
        }

        // 0xBB: G_TEXTURE. The stream sends scale 0 (`BB000001 00000000`), which collapses RT64's
        // texcoords to texel 0; the ucode ignores the scale, so force full scale.
        void texture_f5(State* state, DisplayList** dl) {
            const uint8_t tile = (*dl)->p0(8, 3), level = (*dl)->p0(11, 3), on = (*dl)->p0(0, 8);
            uint16_t sc = (*dl)->p1(16, 16), tc = (*dl)->p1(0, 16);
            if (sc == 0) sc = 0xFFFF;
            if (tc == 0) tc = 0xFFFF;
            state->rsp->setTexture(tile, level, on, sc, tc);
            { static int s_ss2 = -1; if (s_ss2 < 0) { const char* e = std::getenv("ROGUESQ_LOG_SKYSEQ"); s_ss2 = (e && e[0] == '1') ? 1 : 0; }
              const LoadTile &rtt = state->rdp->tiles[tile];
              if (s_ss2 && rtt.fmt == 0 && rtt.siz == 3) { std::fprintf(stderr, "[skyseq] GTEX tile=%u siz=%u on=%u\n", tile, rtt.siz, on); std::fflush(stderr); } }
        }

        // fillRect wrapper kept for the rdpstate module's declaration (op_02 coupling is gone).
        void fillRect_op02_aware(State* state, DisplayList** dl) {
            fillRect_logged(state, dl);
        }

        // ---------------------------------------------------------------- setup
        void setup(GBI* gbi) {
            GBI_F3DEX::setup(gbi);

            // Factor 5 geometry stream.
            gbi->map[0x01] = &op_01_matrix;
            gbi->map[0x02] = &op_02_colors;
            gbi->map[0x03] = &op_03_f5;
            gbi->map[0x04] = &op_04_vertex;
            gbi->map[0x14] = &op_consume16;    // 16-byte state command (not a vertex batch)
            gbi->map[0x13] = &op_13_quad;
            gbi->map[0xBF] = &op_bf_tri;
            gbi->map[0xB4] = &op_b4_quad;
            gbi->map[0xBB] = &texture_f5;

            // Flow.
            gbi->map[0x06] = &op_06_strict_dl;
            gbi->map[0x80] = &op_80_header;
            gbi->map[0xB5] = &op_b5_next_chunk;
            gbi->map[0x07] = &op_07_branch;
            {
                static bool s_op05 = env_on("ROGUESQ_F5_OP05_32", true);
                if (s_op05) gbi->map[0x05] = &op_05_record;
            }

            // Ucode dispatch-table aliases (DMEM 0xD6 table): low opcodes 0x08..0x12 run the
            // same handlers as 0xBF..0xB5 (0x0C = G_TEXTURE follows every 0x05 sprite record).
            gbi->map[0x08] = &op_bf_tri;
            gbi->map[0x09] = &op_consume16;      // BE
            gbi->map[0x0A] = &op_consume16;      // BD
            // G_MW_FOG (index 8) on this ucode is a single 16.16 multiplier for a near-plane fade, not the
            // F3DEX (mul, offset) pair; fed to RT64's z/w curve it whites out the nearest geometry whenever
            // the low half reads negative (the LucasArts flyover "hole"). Hardware shows no distance fog
            // (verified against PJ64 goldens), so keep RT64 fog at zero. ROGUESQ_F5_FOG_RAW=1 restores the
            // F3DEX reading; ROGUESQ_F5_FOG_FORCE="mul,offset" pins fixed values for A/B.
            { static GBIFunction s_mw = gbi->map[0xBC];
              s_mw_orig = s_mw;
              gbi->map[0xBC] = +[](State* state, DisplayList** dl) {
                  static bool s_raw = env_on("ROGUESQ_F5_FOG_RAW", false);
                  static const char* s_force = std::getenv("ROGUESQ_F5_FOG_FORCE");
                  if (((*dl)->w0 & 0xFF) == 8 && !s_raw) {
                      int fm = 0, fo = 0;
                      if (s_force && *s_force) std::sscanf(s_force, "%d,%d", &fm, &fo);
                      state->rsp->setFog((int16_t)fm, (int16_t)fo);
                      return;
                  }
                  s_mw_orig(state, dl);
              }; }
            gbi->map[0x0B] = gbi->map[0xBC];     // moveword
            gbi->map[0x0C] = &texture_f5;        // BB
            gbi->map[0x0D] = gbi->map[0xBA];
            gbi->map[0x0E] = gbi->map[0xB9];
            gbi->map[0x0F] = gbi->map[0xB8];     // endDl
            gbi->map[0x10] = gbi->map[0xB7];
            gbi->map[0x11] = gbi->map[0xB6];
            gbi->map[0x12] = &op_b5_next_chunk;

            // Reused F3DEX bytes that carry Factor 5 data.
            gbi->map[0xB0] = &op_noop;   // F3DEX branch_z: packed data here
            gbi->map[0xB2] = &op_noop;   // F3DEX modify_vtx
            gbi->map[0xAF] = &op_noop;
            gbi->map[0xEF] = &op_noop;
            {
                static bool s_bdbe = env_on("ROGUESQ_F5_BDBE", true);
                static bool s_be16 = env_on("ROGUESQ_F5_BE16", true);
                if (s_bdbe) {
                    gbi->map[0xBD] = s_be16 ? &op_consume16 : &op_noop;
                    gbi->map[0xBE] = s_be16 ? &op_consume16 : &op_noop;
                }
            }

            // RDP-state / raster wrappers (rt64_gbi_f5_rdpstate.cpp).
            gbi->map[0xFC] = &setCombine_logged;
            gbi->map[0xF5] = &setTile_logged;
            gbi->map[0xF6] = &fillRect_op02_aware;
            gbi->map[0xF7] = &setFillColor_overridden;
            gbi->map[0xB9] = &setOtherModeL_logged;
            gbi->map[0xBA] = &setOtherModeH_logged;
            gbi->map[0xED] = &setScissor_logged;
            gbi->map[0xFD] = &setTextureImage_filtered;
            gbi->map[0xF0] = &loadTLUT_guarded;
            gbi->map[0xF3] = &loadBlock_guarded;
            gbi->map[0xF4] = &loadTile_guarded;
            gbi->map[0xFF] = &setColorImage_filtered;
            gbi->map[0xE4] = &texrectLLE_guarded;
            gbi->map[0xE5] = &texrectFlipLLE_guarded;

            // Chunk-bounded fetch around every handler (must be last).
            f5_install_bounded(gbi, std::make_integer_sequence<int, UCODE_MAP_SIZE>{});
        }
    }
}

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
#include <unordered_set>
#include <unordered_map>
#include <cmath>

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

        // ---- Per-object node-id map for frame interpolation (ROGUESQ_F5_NODE_ID) ----
        // The game writes each scene node's world matrix into a double-buffered ring (buffer A
        // 0x80700040+, buffer B 0x80710040+, 0x40-stride submit-order slots). A game-side hook on
        // traverseSceneGraphRecursive's matrix-write site calls rs64_f5_map_node(ringDst, nodeId)
        // per object, so op_01_matrix can stamp the stable scene-node identity as the interpolation
        // matrixId and pair the same object across frames regardless of submit-order shuffle. The
        // map is rebuilt every frame (a buffer's map clears when its slot 0 is written), so it can't
        // accumulate the stale mappings that made the earlier persistent pointer map warp.
        static bool f5_node_id_enabled() {
            static bool s = env_on("ROGUESQ_F5_NODE_ID", false);
            return s;
        }
        // ROGUESQ_F5_VTX_INTERP=1: also interpolate per-vertex positions/texcoords for id-matched
        // objects (AUTO = only when vertex counts match and hashes differ), to smooth F5's vertex-
        // baked motion (terrain LOD morph). SKIP by default (transform-only).
        static bool f5_vtx_interp_enabled() {
            static bool s = env_on("ROGUESQ_F5_VTX_INTERP", false);
            return s;
        }
        // ROGUESQ_F5_TERRAIN_ID=1: id scrolling terrain tiles by quantized world position (stable
        // per world cell across grid re-centers) instead of the reused slot/node pointer.
        static bool f5_terrain_id_enabled() {
            static bool s = env_on("ROGUESQ_F5_TERRAIN_ID", false);
            return s;
        }
        // ROGUESQ_F5_PROJ_LOG=1: log the projection matrix depth terms + derived near/far, and tag each
        // viewport (STREAM from 03 80, or SYNTH from scissor) with its z scale/translate. Diagnoses
        // whether the far plane is squeezed by RT64's depth mapping (the late-level horizontal far cut).
        static bool f5_proj_log() { static bool s = env_on("ROGUESQ_F5_PROJ_LOG", false); return s; }
        // ROGUESQ_F5_ALT_PROBE=1: log modelview translations and loaded-vertex ranges to find what
        // saturates/wraps as the player climbs (ship faces cull progressively with altitude).
        static bool f5_alt_probe() { static bool s = env_on("ROGUESQ_F5_ALT_PROBE", false); return s; }
        // ROGUESQ_F5_FAR_CLAMP=1 (default on): RS64's projection asymptotes ndc.z to ~1.0062 (>1), so
        // distant geometry (view.z beyond ~9700) is discarded by RT64's z>1 clip where the N64 RDP drew
        // it clamped-and-fogged -> a horizontal far cut in high-ceiling levels (Calamari/Sullust/etc).
        // Fix: when a projection's depth asymptote (M[10]) exceeds 1, affine-remap ndc.z to pull the
        // asymptote just under 1 while pinning the near plane (ndc.z=-1), so nothing crosses the clip.
        // Default ON. The projection asymptotes ndc.z to ~1.006, so the tallest structures' tops sit
        // past the far plane and get GPU-clipped (N64 clamped them instead). The remap pulls the
        // asymptote under 1 so they draw. Its target T must sit well BELOW the skybox dome's pinned
        // depth (0.99999, rt64_state.cpp) or the tops z-fight the dome into shards.
        // Default OFF: superseded by the N64-style far-plane depth CLAMP in RSP::setVertex (rt64_rsp.cpp).
        // The affine remap preserved depth ORDER past the far plane, so the horizon haze overlay (farther)
        // always lost the depth test to beyond-far water instead of tying with it as on hardware -> the
        // haze was cut along the water's far row (the horizon "ring"). Kept env-gated for A/B only.
        static bool f5_far_clamp() { static bool s = env_on("ROGUESQ_F5_FAR_CLAMP", false); return s; }
        static constexpr uint32_t F5_NODE_MAP_SLOTS = 256;
        static uint32_t s_node_map[2][F5_NODE_MAP_SLOTS] = {};   // [buffer][slot] = stamped id (0 = leave AUTO)
        // A node only gets an interpolation id if it existed last frame (stable). Transient nodes
        // (per-frame terrain tessellation, just-spawned objects) are not paired by id — they fall
        // back to AUTO geometric matching, which handles deforming terrain far better than a
        // never-matching linear id (that would drop interpolation and stutter).
        static std::unordered_set<uint32_t> s_nodes_prev, s_nodes_cur;
        // Per-buffer histogram of drawable pointers (renderNode = *(node+0x10); drawable = renderNode+0x08).
        // A drawable used by exactly one node this frame = a distinct model instance (ship) that benefits
        // from stable-id interpolation; a shared drawable = instanced particles / terrain tiles that
        // flicker or morph under it. Rebuilt at each frame's first slot. Requires RDRAM (op_01 side).
        static std::unordered_map<uint32_t, int> s_draw_hist[2];
        static uint32_t f5_node_drawable(const uint8_t* ram, uint32_t node) {
            const uint32_t rn = rd_be_u32(ram, node + 0x10);
            return (rn >= 0x80000000u) ? rd_be_u32(ram, rn + 0x08) : 0u;
        }
        static void f5_rebuild_draw_hist(const uint8_t* ram, int buf) {
            s_draw_hist[buf].clear();
            for (uint32_t i = 0; i < F5_NODE_MAP_SLOTS; ++i) {
                const uint32_t node = s_node_map[buf][i];
                if (node < 0x80000000u) continue;
                const uint32_t dr = f5_node_drawable(ram, node);
                if (dr >= 0x80000000u) ++s_draw_hist[buf][dr];
            }
        }
        static inline int f5_ring_buffer(uint32_t off) { return (off & 0x10000u) ? 1 : 0; }
        static inline int f5_ring_slot(uint32_t off) {
            const uint32_t rel = (off & 0xFFFFu);
            if (rel < 0x40u) return -1;
            const uint32_t s = (rel - 0x40u) / 0x40u;
            return s < F5_NODE_MAP_SLOTS ? (int)s : -1;
        }
        void f5_map_node_impl(uint32_t ringDst, uint32_t nodeId) {
            const uint32_t off = ringDst & 0x00FFFFFFu;
            if (off < 0x700000u || off >= 0x720000u) return;
            const int buf = f5_ring_buffer(off);
            const int slot = f5_ring_slot(off);
            if (slot < 0) return;
            if (slot == 0) {   // slot 0 = start of a new frame (one per frame, in the active buffer)
                for (uint32_t i = 0; i < F5_NODE_MAP_SLOTS; ++i) s_node_map[buf][i] = 0;
                s_nodes_prev.swap(s_nodes_cur);
                s_nodes_cur.clear();
            }
            s_nodes_cur.insert(nodeId);
            // Store the raw node pointer for nodes present last frame (stable); 0 = leave AUTO.
            // op_01 reads the node's flags via RDRAM to classify before stamping an id.
            s_node_map[buf][slot] = s_nodes_prev.count(nodeId) ? nodeId : 0u;
        }
        // Look up the id a hook stored for this ring address; 0 if none.
        static uint32_t f5_lookup_node_id(uint32_t w1) {
            const uint32_t off = w1 & 0x00FFFFFFu;
            if (off < 0x700000u || off >= 0x720000u) return 0;
            const int buf = f5_ring_buffer(off);
            const int slot = f5_ring_slot(off);
            if (slot < 0) return 0;
            return s_node_map[buf][slot];
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

        // ROGUESQ_DROP_PROBE: log dropped/no-op'd F5 commands (opcode + payload + a few following
        // words), deduped by opcode. If a particle/sprite batch command is being silently consumed,
        // it shows here with structured (non-filler) payload.
        static void drop_log(const char* who, DisplayList** dl) {
            static bool s_on = env_on("ROGUESQ_DROP_PROBE", false);
            if (!s_on) return;
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            const uint8_t op = (uint8_t)(w0 >> 24);
            static std::unordered_map<uint8_t, uint32_t> s_cnt;
            uint32_t& c = s_cnt[op]; ++c;
            if (c <= 3 || (c % 2000) == 0) {
                std::fprintf(stderr, "[drop %s] op=%02X cnt=%u w0=%08X w1=%08X next=[%08X %08X %08X %08X]\n",
                    who, op, c, w0, w1, (*dl)[1].w0, (*dl)[1].w1, (*dl)[2].w0, (*dl)[2].w1);
                std::fflush(stderr);
            }
        }
        void op_noop(State*, DisplayList** dl) { drop_log("noop", dl); }

        // Skip the 8-byte payload of a 16-byte command.
        void op_consume16(State*, DisplayList** dl) { drop_log("c16", dl); (*dl)++; }
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
        // True once the game sent G_SETPRIMDEPTH (0xEE) for the current zSource=PRIM state (the sky
        // pins itself to 0x7FBE this way). Cleared when a 0xB9 turns zSource off. Effect meshes set
        // zSource=PRIM without an 0xEE and rely on the synthesized per-face depth instead.
        static bool s_prim_depth_from_game = false;
        static uint32_t s_task_budget_trip = 0;
        static uint32_t s_task_tiles = 0;
        static uint32_t s_task_maxchain = 0;
        static void f5_ensure_viewport(State* state);
        // Above the N64's 8 MB (the recomp heap does use 0x71E000; hardware never sees this range; RT64's
        // segmented mask allows 16 MB and the host buffer is 512 MB).
        static constexpr uint32_t F5_VTX_SCRATCH = 0x00A00000u;   // 256 x 16 bytes
        static constexpr uint32_t F5_VP_SCRATCH  = 0x00A01000u;
        static constexpr uint32_t F5_PROJ_SCRATCH = 0x00A01100u;  // 16 host floats: remapped projection
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
            if (n < (uint32_t)F5_CHAIN_MAX) { s_chain[depth][n++] = chunk; if (n > s_task_maxchain) s_task_maxchain = n; }
            return false;
        }
        static void f5_next_chunk(State* state, DisplayList** dl) {
            const uint32_t d = f5_depth(state);
            const uint32_t base = (d < (uint32_t)F5_MAX_DEPTH) ? s_chunk_base[d] : 0;
            const uint32_t next = base ? f5_rd32(state, base) : 0;
            const uint32_t target = next & 0x00FFFFFFu;
            // The allocated chunk list is doubly linked: the next chunk's prev word must point back here.
            // A stale call into a recycled chunk otherwise walks the whole free list (thousands of garbage faces).
            const bool fwdValid = (next >> 24) == 0x80u && target != 0 && target + 0x108u <= RDRAMSize;
            const bool backLinked = fwdValid && f5_rd32(state, target + 4) == (0x80000000u | base);
            // ROGUESQ_F5_CHUNK_LINK_LOOSE: follow a forward-valid link even when the back-link hasn't been
            // written yet. In Release the menu overlay chunk's prev word races the gfx walk, so backLinked
            // is intermittently false and the strict rule ends the DL -> the medal/insignia/preview draws are
            // dropped. The per-task entry/face budgets + revisit detection remain the free-list-storm backstop.
            static const bool s_linkLoose = env_on("ROGUESQ_F5_CHUNK_LINK_LOOSE", false);
            const bool linked = backLinked || (s_linkLoose && fwdValid);
            // ROGUESQ_F5_LINK_PROBE: log the forward-valid-but-back-link-mismatch case (the overlay-drop signature).
            { static const bool s_lp = env_on("ROGUESQ_F5_LINK_PROBE", false); static int s_lpn = 0;
              if (s_lp && fwdValid && !backLinked && ++s_lpn <= 24) {
                  std::fprintf(stderr, "[f5-link] fwd-valid but bad back-link: base=%06X target=%06X prev=%08X (want %08X)\n",
                      base, target, f5_rd32(state, target + 4), 0x80000000u | base); std::fflush(stderr); } }
            // Backstop for a stale call into a recycled chunk walking the (also doubly linked) free list.
            // Cap = F5_CHAIN_MAX, the window f5_chunk_revisit can still detect a cycle in. A dense frame
            // legitimately chains ~84 chunks (LucasArts flyover, 359 tiles / 1597 faces); the old cap of 64
            // truncated those lists mid-frame and dropped a contiguous block of terrain.
            static const uint32_t s_chain_cap = [](){ const char* e = std::getenv("ROGUESQ_F5_CHAIN_CAP"); return (e && *e) ? (uint32_t)std::atoi(e) : (uint32_t)F5_CHAIN_MAX; }();
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
                  if (s_tl) { std::fprintf(stderr, "[f5-task] done before #%llu: faces=%u entries=%u hops=%u budget_trip=%u tiles=%u maxchain=%u\n",
                      (unsigned long long)s_chunk_counter, s_task_faces, s_task_entries, s_chunk_hops, s_task_budget_trip, s_task_tiles, s_task_maxchain); std::fflush(stderr); } }
                s_task_budget_trip = 0; s_task_tiles = 0; s_task_maxchain = 0;
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
                // y is h<<4 (up to +-524k); a bare int16 cast wraps for source heights > 2047 (tall
                // high-ceiling terrain) and snaps the vertex to the bottom -> terrain holes. Saturate.
                { const int32_t yy = y + h[k]; tmp[k].y = (int16_t)(yy > 32767 ? 32767 : (yy < -32768 ? -32768 : yy)); }
                tmp[k].x = (int16_t)px[k]; tmp[k].z = (int16_t)pz[k]; tmp[k].flag = 0;
                tmp[k].s = us[k]; tmp[k].t = vt[k];
                tmp[k].color.r = (uint8_t)(col[k] >> 24); tmp[k].color.g = (uint8_t)(col[k] >> 16);
                tmp[k].color.b = (uint8_t)(col[k] >> 8);  tmp[k].color.a = (uint8_t)col[k];
            }
            state->rsp->setVertex(0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)), 4, F5_FACE_SLOT);
            state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 3, F5_FACE_SLOT + 2);
            state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 1, F5_FACE_SLOT + 3);
            s_task_faces += 2;
        }

        // Emit N terrain strips (2*(N+1) verts each, laid out contiguously) exactly as the
        // inline grid loop did: one setVertex + two drawIndexedTri per strip.
        static void f5_emit_terrain_strips(State* state, const RSP::Vertex* verts, int N) {
            RSP::Vertex* tmp = reinterpret_cast<RSP::Vertex*>(
                state->fromRDRAM(F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)));
            const uint32_t base = 0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex));
            const int stripVerts = 2 * (N + 1);
            for (int rr = 0; rr < N; ++rr) {
                const RSP::Vertex* strip = verts + (size_t)rr * stripVerts;
                for (int i = 0; i < stripVerts; ++i) tmp[i] = strip[i];
                state->rsp->setVertex(base, stripVerts, F5_FACE_SLOT);
                for (int cc = 0; cc < N; ++cc) {
                    const int v0 = F5_FACE_SLOT + cc, v2 = F5_FACE_SLOT + (N + 1) + cc;
                    state->rsp->drawIndexedTri(v0, v2 + 1, v2);
                    state->rsp->drawIndexedTri(v0, v0 + 1, v2 + 1);
                }
            }
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
            // Parser overlay 0xC samples the N*N input at the LOD stride: output (i,j) = input
            // [(i*stride)*N + j*stride], an M*M grid. Samples between stride points are never read at
            // this LOD and hold unrelated heights -- meshing them produced spikes on steep terrain.
            const int stride = 1 << shift;
            float G[5][5] = {};
            for (int i = 0; i < M && i < 5; ++i) for (int j = 0; j < M && j < 5; ++j)
                G[i][j] = (float)(int8_t)ram[(hptr + (uint32_t)((i * stride) * gridN + j * stride)) ^ 3];
            // Edge LOD seams (overlays 0x14 rows, 0x18 columns). Record bytes +4..+7 are the left, right,
            // top and bottom neighbour LODs. When one differs from this tile's shift, the odd samples along
            // that edge become the mean of their neighbours (a straight edge, no T-junction), then blend
            // toward the corner line by that edge's weight (+0x16 L, +0x18 R, +0x1A T, +0x1C B, 16.16).
            {
                static bool s_seam = env_on("ROGUESQ_F5_TERRAIN_SEAMS", true);
                const uint32_t nbw = rec[0].w1;
                const int nb[4] = { (int)((nbw >> 24) & 0xFF), (int)((nbw >> 16) & 0xFF), (int)((nbw >> 8) & 0xFF), (int)(nbw & 0xFF) };
                const float wt[4] = { (float)(rec[2].w1 & 0xFFFFu) / 65536.0f, (float)((rec[3].w0 >> 16) & 0xFFFFu) / 65536.0f,
                                      (float)(rec[3].w0 & 0xFFFFu) / 65536.0f, (float)((rec[3].w1 >> 16) & 0xFFFFu) / 65536.0f };
                // edge e: 0=L (col 0), 1=R (col M-1), 2=T (row 0), 3=B (row M-1); k walks along the edge
                auto at = [&](int e, int k) -> float& {
                    return e == 0 ? G[k][0] : e == 1 ? G[k][M - 1] : e == 2 ? G[0][k] : G[M - 1][k]; };
                for (int e = 0; s_seam && M > 2 && e < 4; ++e) {
                    if (nb[e] == shift) continue;
                    for (int k = 1; k + 1 < M; k += 2) at(e, k) = 0.5f * (at(e, k - 1) + at(e, k + 1));
                    if (wt[e] > 0.0f) {
                        const float c0 = at(e, 0), c1 = at(e, M - 1);
                        for (int k = 1; k + 1 < M; ++k) {
                            const float line = c0 + (c1 - c0) * (float)k / (float)(M - 1);
                            at(e, k) = line * wt[e] + at(e, k) * (1.0f - wt[e]);
                        }
                    }
                }
            }
            // Upsample the M*M grid onto the 5x5 the mesh is built from (identity for M == 5).
            float H[25];
            for (int r = 0; r < 5; ++r) for (int c = 0; c < 5; ++c) {
                const float gi = (float)r * (float)(M - 1) / 4.0f, gj = (float)c * (float)(M - 1) / 4.0f;
                int i0 = (int)gi, j0 = (int)gj; if (i0 > M - 2) i0 = M - 2; if (j0 > M - 2) j0 = M - 2;
                if (i0 < 0) i0 = 0; if (j0 < 0) j0 = 0;
                const int i1 = (M > 1) ? i0 + 1 : i0, j1 = (M > 1) ? j0 + 1 : j0;
                const float ti = gi - (float)i0, tj = gj - (float)j0;
                H[r * 5 + c] = (G[i0][j0] * (1.0f - tj) + G[i0][j1] * tj) * (1.0f - ti)
                             + (G[i1][j0] * (1.0f - tj) + G[i1][j1] * tj) * ti;
            }
            // ROGUESQ_LOG_GFX_TASK: trace near-LOD (shift 0) and blended tiles to catch transition garbage.
            { static bool s_lg = env_on("ROGUESQ_LOG_GFX_TASK", false); static int s_n = 0;
              const uint32_t w5 = rec[2].w1 & 0xFFFF, w6 = rec[3].w0 & 0xFFFF;
              bool blank = true; for (int k = 0; k < 25 && blank; ++k) blank = (H[k] == 0.0f);
              if (s_lg && (shift == 0 || w5 || w6 || blank) && ++s_n <= 600) {
                  int hmin = 127, hmax = -128; for (int k = 0; k < 25; ++k) { if (H[k] < hmin) hmin = H[k]; if (H[k] > hmax) hmax = H[k]; }
                  std::fprintf(stderr, "[f5-tile] task=%llu x=%d y=%d z=%d size=%d N=%d shift=%d w5=%04X w6=%04X hptr=%06X h=[%d,%d] blank=%d" "\n",
                      (unsigned long long)state->displayListCounter, x, y, z, sz, gridN, shift, w5, w6, hptr, hmin, hmax, (int)blank); std::fflush(stderr); } }
            // Subdivision: the ucode (overlay 0x14) subdivides the 5x5 into a finer interpolated mesh for
            // smooth slopes; raw 5x5 quads look faceted. Bilinear-interpolate to (4*S+1)^2 verts. S from env
            // (per-tile LOD lives in w1 low bytes; uniform S avoids inter-tile cracks). Grid tiles span (M-1)*size
            // (step5 per 5x5 cell); UV spans the tile over the 4 cells; colors are read at the hardware stride
            // (real values at even samples, odd are checkerboard filler) so sample the nearest even cell.
            // Distance LOD: the game's own per-tile `shift` is its distance band (0 = near, >=1 = far).
            // Near tiles keep full subdivision (smooth foreground); far tiles use fewer strips, which is
            // where the frame-hitch cost lives (op_05 emission scales with strip count). Tile edges are
            // linearly interpolated, so a finer tile's edge verts stay collinear with a coarser neighbour
            // -- no geometric crack across an LOD boundary.
            static int s_sub_near = -1, s_sub_far = -1;
            if (s_sub_near < 0) {
                const char* v = std::getenv("ROGUESQ_F5_TERRAIN_SUB"); s_sub_near = (v && *v) ? std::atoi(v) : 2;
                if (s_sub_near < 1) s_sub_near = 1; if (s_sub_near > 4) s_sub_near = 4;
                const char* vf = std::getenv("ROGUESQ_F5_TERRAIN_SUB_FAR"); s_sub_far = (vf && *vf) ? std::atoi(vf) : 1;
                if (s_sub_far < 1) s_sub_far = 1; if (s_sub_far > 4) s_sub_far = 4;
            }
            const int s_sub = (shift == 0) ? s_sub_near : s_sub_far;
            const int N = 4 * s_sub;   // cells per axis; N+1 verts per axis; 2*(N+1) <= 34 slots for S<=4
            auto emit_vertex = [&](RSP::Vertex& v, float fx, float fy) {
                int x0 = (int)fx, y0 = (int)fy; if (x0 > 3) x0 = 3; if (y0 > 3) y0 = 3;
                const float tx = fx - x0, ty = fy - y0;
                const float h = (H[y0*5+x0]*(1-tx) + H[y0*5+x0+1]*tx) * (1-ty)
                              + (H[(y0+1)*5+x0]*(1-tx) + H[(y0+1)*5+x0+1]*tx) * ty;
                v.x = (int16_t)(x + (int32_t)(fx * step5));
                { const int32_t yy = y + (int32_t)(h * hcoeff); v.y = (int16_t)(yy > 32767 ? 32767 : (yy < -32768 ? -32768 : yy)); }   // saturate, see f5_tile_quad
                v.z = (int16_t)(z + (int32_t)(fy * step5));
                v.flag = 0; v.s = (int16_t)(uvcell * fx); v.t = (int16_t)(uvcell * (4.0f - fy));
                // Colors: real samples sit every (1<<shift) in the N*N grid; the parser overlay averages
                // between them, so interpolate bilinearly instead of snapping to the nearest sample.
                {
                    const int cstep = 1 << shift;
                    const float gx = fx * (float)(gridN - 1) / 4.0f, gy = fy * (float)(gridN - 1) / 4.0f;
                    int x0 = ((int)gx / cstep) * cstep, y0 = ((int)gy / cstep) * cstep;
                    if (x0 > gridN - 1) x0 = gridN - 1; if (y0 > gridN - 1) y0 = gridN - 1;
                    const int x1 = (x0 + cstep <= gridN - 1) ? x0 + cstep : x0, y1 = (y0 + cstep <= gridN - 1) ? y0 + cstep : y0;
                    const float tx = (x1 != x0) ? (gx - (float)x0) / (float)(x1 - x0) : 0.0f;
                    const float ty = (y1 != y0) ? (gy - (float)y0) / (float)(y1 - y0) : 0.0f;
                    auto ch = [&](int cx, int cy, int k) -> float { return (float)ram[(cptr + (uint32_t)(cy * gridN + cx) * 4 + (uint32_t)k) ^ 3]; };
                    uint8_t out[4];
                    for (int k = 0; k < 4; ++k) {
                        const float top = ch(x0, y0, k) * (1.0f - tx) + ch(x1, y0, k) * tx;
                        const float bot = ch(x0, y1, k) * (1.0f - tx) + ch(x1, y1, k) * tx;
                        float val = top * (1.0f - ty) + bot * ty + 0.5f;
                        out[k] = (uint8_t)(val > 255.0f ? 255.0f : val);
                    }
                    v.color.r = out[0]; v.color.g = out[1]; v.color.b = out[2]; v.color.a = out[3];
                }
            };
            std::vector<RSP::Vertex> geom((size_t)N * 2 * (N + 1));
            for (int rr = 0; rr < N; ++rr) {   // one 2-row strip at a time
                RSP::Vertex* strip = geom.data() + (size_t)rr * 2 * (N + 1);
                for (int cc = 0; cc <= N; ++cc) {
                    emit_vertex(strip[cc],           (float)cc / s_sub, (float)rr / s_sub);
                    emit_vertex(strip[(N + 1) + cc], (float)cc / s_sub, (float)(rr + 1) / s_sub);
                }
            }
            f5_emit_terrain_strips(state, geom.data(), N);
        }

        void op_05_record(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0;
            // ROGUESQ_LOG_GFX_TASK: report op-05 headers whose byte1 (samples per row) is not 5, the only
            // record shape the parser below handles; anything else is walked as an 8-byte command.
            { static bool s_lg = env_on("ROGUESQ_LOG_GFX_TASK", false); static int s_n = 0;
              if (s_lg && ((w0 >> 16) & 0xFFu) != 0x05u && ++s_n <= 40) {
                  std::fprintf(stderr, "[f5-op05] unusual header %08X %08X next %08X %08X\n", w0, (*dl)->w1, (*dl)[1].w0, (*dl)[1].w1); std::fflush(stderr); } }
            if (((w0 >> 16) & 0xFFu) != 0x05u) return;      // plain 8-byte command
            ++s_task_tiles;
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
        // Fog = two signed 16.16 words: M at DMEM 0x160 (moveword 8 or 03 88) and O at DMEM 0x164
        // (moveword 0x0A or 03 88). The ucode stores alpha = 255 * clamp(depth*M + O, 0, 1). RT64 takes
        // alpha = depth*mul + offset, so mul = 255*M, offset = 255*O, written as floats: the values
        // exceed RT64's int16 setFog range (a ramp only ~0.001 of depth wide).
        static double s_f5_fog_m = 0.0, s_f5_fog_o = 0.0;
        static void f5_apply_fog(State* state) {
            static int s_nofog = -1;
            if (s_nofog < 0) { const char* e = std::getenv("ROGUESQ_F5_NO_FOG"); s_nofog = (e && e[0] && e[0] != '0') ? 1 : 0; }
            if (s_nofog) { state->rsp->fog.mul = 0.0f; state->rsp->fog.offset = 0.0f; state->rsp->fogChanged = true; return; }
            if (f5_proj_log()) { static int n = 0; if ((n++ % 120) == 0)
                std::fprintf(stderr, "[f5-fog] M=%.6f O=%.6f -> mul=%.3f offset=%.3f\n",
                    s_f5_fog_m, s_f5_fog_o, 255.0 * s_f5_fog_m, 255.0 * s_f5_fog_o); std::fflush(stderr); }
            state->rsp->fog.mul = (float)(255.0 * s_f5_fog_m);
            state->rsp->fog.offset = (float)(255.0 * s_f5_fog_o);
            state->rsp->fogChanged = true;
        }
        static void f5_set_viewport(State* state, const int16_t* vs, const int16_t* vt);
        void op_03_f5(State* state, DisplayList** dl) {
            const uint32_t sub = ((*dl)->w0 >> 16) & 0xFF;
            if (sub == 0x80) {
                const uint32_t a = (*dl)[1].w0, b = (*dl)[1].w1, c = (*dl)[2].w0, d = (*dl)[2].w1;
                const int16_t vs[4] = { (int16_t)(a >> 16), (int16_t)a, (int16_t)(b >> 16), (int16_t)b };
                const int16_t vt[4] = { (int16_t)(c >> 16), (int16_t)c, (int16_t)(d >> 16), (int16_t)d };
                f5_set_viewport(state, vs, vt);
            } else if (sub == 0x88) {
                s_f5_fog_m = (double)(int32_t)(*dl)[1].w0 / 65536.0;
                s_f5_fog_o = (double)(int32_t)(*dl)[1].w1 / 65536.0;
                f5_apply_fog(state);
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
            if (f5_proj_log()) { static int n = 0; if ((n++ % 120) == 0)
                std::fprintf(stderr, "[f5-vp] STREAM vscale[%d %d %d %d] vtrans[%d %d %d %d]\n",
                    vp->vscale[0], vp->vscale[1], vp->vscale[2], vp->vscale[3],
                    vp->vtrans[0], vp->vtrans[1], vp->vtrans[2], vp->vtrans[3]); std::fflush(stderr); }
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
            if (f5_proj_log()) { static int n = 0; if ((n++ % 120) == 0)
                std::fprintf(stderr, "[f5-vp] SYNTH w=%d h=%d vscale[%d %d %d %d] vtrans[%d %d %d %d]\n", w, h,
                    vp->vscale[0], vp->vscale[1], vp->vscale[2], vp->vscale[3],
                    vp->vtrans[0], vp->vtrans[1], vp->vtrans[2], vp->vtrans[3]); std::fflush(stderr); }
        }

        // 0x01: matrix load. byte1 0x03 = projection, 0x02 = modelview; a matrix whose bottom-right
        // element is 0 with a non-zero [2][3] is a projection whatever the byte says.
        // ROGUESQ_F5_CULL_DIST=<units>: RT64-side per-object distance cull. op_01_matrix sets s_cull_skip
        // when a model's camera-space origin (modelview translation) exceeds the threshold; the geometry
        // emitters then drop its setVertex/drawIndexedTri, cutting render-thread draw-recording in dense
        // scenes. Layered on the game's own LOD, so keep the threshold far to avoid popping submitted
        // objects. Terrain (op_05, no 0x01) and effect billboards (op_bd) are unaffected. 0 disables.
        static bool s_cull_skip = false;
        static float f5_cull_dist2() {
            static float d2 = -1.0f;
            if (d2 < -0.5f) { const char* e = std::getenv("ROGUESQ_F5_CULL_DIST"); float d = (e && e[0]) ? (float)std::atof(e) : 0.0f; d2 = (d > 0.0f) ? d * d : 0.0f; }
            return d2;
        }

        void op_01_matrix(State* state, DisplayList** dl) {
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            const uint32_t addr = state->rsp->fromSegmentedMasked(w1);
            if (addr + 64 > RDRAMSize) return;
            const uint8_t* ram = state->RDRAM;
            const int m33 = rd_be_s16(ram, w1 + 2 * 15), m23 = rd_be_s16(ram, w1 + 2 * 11);
            const bool proj = (((w0 >> 16) & 0xFF) == 0x03) || (m33 == 0 && m23 != 0);
            if (!f5_native_active()) return;
            f5_ensure_viewport(state);

            // Per-object distance cull: evaluate on the modelview load that delimits each object.
            {
                const float thr2 = f5_cull_dist2();
                if (thr2 > 0.0f && !proj) {
                    const float tx = (float)rd_be_s16(ram, w1 + 2 * 12) + (float)rd_be_u16(ram, w1 + 32 + 2 * 12) / 65536.0f;
                    const float ty = (float)rd_be_s16(ram, w1 + 2 * 13) + (float)rd_be_u16(ram, w1 + 32 + 2 * 13) / 65536.0f;
                    const float tz = (float)rd_be_s16(ram, w1 + 2 * 14) + (float)rd_be_u16(ram, w1 + 32 + 2 * 14) / 65536.0f;
                    s_cull_skip = (tx * tx + ty * ty + tz * tz) > thr2;
                    { static bool s_lg = env_on("ROGUESQ_F5_CULL_LOG", false); static int s_n = 0;
                      if (s_lg && (++s_n & 31) == 0) { std::fprintf(stderr, "[f5-cull] dist=%.0f thr=%.0f skip=%d\n",
                          std::sqrt(tx*tx+ty*ty+tz*tz), std::sqrt(thr2), (int)s_cull_skip); std::fflush(stderr); } }
                } else {
                    s_cull_skip = false;
                }
            }

            // ROGUESQ_F5_SLOT_ID (default off): stamp each per-object modelview with a frame-stable
            // matrixId so RT64 pairs the same object across frames by id (ORDER_LINEAR) instead of
            // by geometry. The object's model*view matrix is written into a double-buffered ring
            // (0x80700040 even frames / 0x80710040 odd, 0x40-stride submit-order slots); the low 16
            // bits of the address are identical in both buffers, so (w1 & 0xFFFF) is the slot id with
            // no buffer-flip and no persistent state to go stale. Recomputed every frame from the
            // live stream, so it self-corrects on spawn/despawn instead of accumulating mispairs.
            if (!proj) {
                  // Prefer the stable scene-node id (populated by the game hook) when enabled; fall
                  // back to the submit-order slot id (ROGUESQ_F5_SLOT_ID, experiment) otherwise.
                  uint32_t id = 0;
                  bool isTerrain = false;
                  if (f5_node_id_enabled()) {
                      const uint32_t nodePtr = f5_lookup_node_id(w1);   // raw node ptr, or 0
                      if (nodePtr >= 0x80000000u) {
                          // Default stamps every stable node (ships smooth; the best result found).
                          // ROGUESQ_F5_UNIQUE_ONLY=1 restricts to nodes with a unique drawable this
                          // frame — excludes particle templates and terrain tiles, but also drops
                          // formation ships (identical model = shared drawable), so it is opt-in.
                          static int s_uniq = -1; if (s_uniq < 0) { const char* e = std::getenv("ROGUESQ_F5_UNIQUE_ONLY"); s_uniq = (e && e[0] == '1') ? 1 : 0; }
                          bool stamp = true;
                          if (s_uniq) {
                              const uint32_t off = w1 & 0x00FFFFFFu;
                              const int buf = (off & 0x10000u) ? 1 : 0;
                              if ((off & 0xFFFFu) == 0x40u) f5_rebuild_draw_hist(state->RDRAM, buf);
                              const uint32_t dr = f5_node_drawable(state->RDRAM, nodePtr);
                              auto it = s_draw_hist[buf].find(dr);
                              stamp = (dr >= 0x80000000u) && (it != s_draw_hist[buf].end()) && (it->second == 1);
                          }
                          if (stamp) id = 0x20000000u | (nodePtr & 0x00FFFFFFu);
                          // ROGUESQ_F5_TERRAIN_ID: terrain tiles are a scrolling pool — the node ptr is
                          // the reused SLOT, which makes RT64 interpolate a slot through the world-cell
                          // change at a grid re-center (the periodic terrain hitch). Instead, id a
                          // terrain tile by its WORLD position (node+0x40 X/Z, quantized) so the draw
                          // showing a given world cell keeps the same id across frames regardless of
                          // slot, and RT64 interpolates its (camera-relative) transform smoothly.
                          // Terrain tiles = shared drawable + a real world translation (effects have ~0).
                          if (f5_terrain_id_enabled()) {
                              const uint32_t offb = w1 & 0x00FFFFFFu; const int bufb = (offb & 0x10000u) ? 1 : 0;
                              if ((offb & 0xFFFFu) == 0x40u) f5_rebuild_draw_hist(state->RDRAM, bufb);
                              const uint32_t dr = f5_node_drawable(state->RDRAM, nodePtr);
                              auto it = s_draw_hist[bufb].find(dr);
                              const bool shared = (it != s_draw_hist[bufb].end()) && (it->second > 1);
                              if (shared) {
                                  auto rf = [&](uint32_t o){ union{uint32_t u;float f;}c; c.u = rd_be_u32(state->RDRAM, nodePtr + o); return c.f; };
                                  float wx = rf(0x40), wz = rf(0x48);
                                  if (fabsf(wx) + fabsf(wz) > 1.0f) {   // real world position => terrain, not an effect at ~origin
                                      // The tile position is relative to a root that tracks the camera, so it shifts
                                      // together at each grid re-center. Add the camera world position (camera struct
                                      // @ MEM[0x80138D1C], horizontal pos +0x4C/+0x50) to get an absolute world cell
                                      // that survives the re-center. ROGUESQ_F5_TERRAIN_ABS=0 disables the add (A/B).
                                      static const float cell = [](){ const char* e = std::getenv("ROGUESQ_F5_TERRAIN_CELL"); float f = (e && e[0]) ? (float)atof(e) : 4.0f; return f > 0.01f ? f : 4.0f; }();
                                      int cx = (int)lroundf(wx / cell), cz = (int)lroundf(wz / cell);
                                      // Add the camera's world position STEPPED to the cell grid (the root re-centers in
                                      // discrete cell steps, so use round(cam/cell) not the continuous position, or a
                                      // per-frame sub-cell fraction makes the id jitter). Yields an absolute cell that
                                      // survives the re-center. ROGUESQ_F5_TERRAIN_ABS=0 disables (A/B).
                                      static int s_abs = -1; if (s_abs < 0) { const char* e = std::getenv("ROGUESQ_F5_TERRAIN_ABS"); s_abs = (e && e[0] == '0') ? 0 : 1; }
                                      if (s_abs) {
                                          const uint32_t camPtr = rd_be_u32(state->RDRAM, 0x80138D1Cu);
                                          if (camPtr >= 0x80000000u) {
                                              union{uint32_t u;float f;}cx2,cz2; cx2.u = rd_be_u32(state->RDRAM, camPtr + 0x4Cu); cz2.u = rd_be_u32(state->RDRAM, camPtr + 0x50u);
                                              cx += (int)lroundf(cx2.f / cell); cz += (int)lroundf(cz2.f / cell);
                                          }
                                      }
                                      id = 0x60000000u | (((uint32_t)cx & 0xFFFu) << 12) | ((uint32_t)cz & 0xFFFu);
                                      isTerrain = true;
                                  }
                              }
                          }
                      }
                  }
                  if (id == 0) {
                      static int s_slotid = -1; if (s_slotid < 0) { const char* e = std::getenv("ROGUESQ_F5_SLOT_ID"); s_slotid = (e && e[0] == '1') ? 1 : 0; }
                      const uint32_t off = w1 & 0x00FFFFFFu;
                      if (s_slotid && off >= 0x700000u && off < 0x720000u) id = 0x00010000u | (w1 & 0xFFFFu);
                  }
                  if (id != 0) {   // never 0 (IGNORE) or 0xFFFFFFFF (AUTO)
                      // Terrain geomorph: the PC port interpolates terrain vertices into position (the N64
                      // snapped). Now that terrain cells have a stable id, enable vertex interpolation for
                      // them so RT64 blends the right cell's vertices across frames. Ships stay transform-
                      // only. ROGUESQ_F5_VTX_INTERP forces it on for everything (A/B).
                      const uint8_t vcomp = (isTerrain || f5_vtx_interp_enabled()) ? G_EX_COMPONENT_AUTO : G_EX_COMPONENT_SKIP;
                      state->rsp->matrixId(id, /*push*/false, /*proj*/false, /*decompose*/true,
                          G_EX_COMPONENT_INTERPOLATE, G_EX_COMPONENT_INTERPOLATE, G_EX_COMPONENT_INTERPOLATE,
                          G_EX_COMPONENT_INTERPOLATE, G_EX_COMPONENT_INTERPOLATE, /*vpos*/vcomp,
                          /*vtc*/vcomp, /*tile*/G_EX_COMPONENT_AUTO, /*lookat*/G_EX_COMPONENT_AUTO,
                          /*order*/G_EX_ORDER_LINEAR, /*aspect*/G_EX_ASPECT_AUTO, /*editable*/G_EX_EDIT_NONE,
                          /*idIsAddress*/false, /*editGroup*/false);
                  }
            }

            bool submittedRemap = false;
            if (proj && f5_far_clamp()) {
                float M[16];
                for (int i = 0; i < 16; ++i) M[i] = (float)rd_be_s16(ram, w1 + 2 * i) + (float)rd_be_u16(ram, w1 + 32 + 2 * i) / 65536.0f;
                const float A = M[10];   // ndc.z asymptote (row-vector: clip.z=col2, clip.w=col3 with M[11]=1, M[15]=0)
                if (A > 1.0001f) {
                    // Affine remap ndc.z' = a*ndc.z + b: pin the near plane (ndc.z=-1 -> -1) and pull the
                    // asymptote A down to T (<1). clip.z' = a*clip.z + b*clip.w => column 2 := a*col2 + b*col3.
                    const float T = 0.99500f;   // keep clear of the dome pinned at 0.99999
                    const float a = (T + 1.0f) / (A + 1.0f), b = a - 1.0f;
                    M[2]  = a * M[2]  + b * M[3];
                    M[6]  = a * M[6]  + b * M[7];
                    M[10] = a * M[10] + b * M[11];
                    M[14] = a * M[14] + b * M[15];
                    float* dst = reinterpret_cast<float*>(state->fromRDRAM(F5_PROJ_SCRATCH));
                    for (int i = 0; i < 16; ++i) dst[i] = M[i];
                    state->rsp->matrixFloat(0x80000000u | F5_PROJ_SCRATCH, 0x03);
                    submittedRemap = true;
                    if (f5_proj_log()) { static int n = 0; if ((n++ % 120) == 0)
                        std::fprintf(stderr, "[f5-remap] FIRED A=%.4f a=%.5f b=%.5f newAsym=%.5f\n",
                            A, a, b, M[10]); std::fflush(stderr); }
                }
            }
            if (!submittedRemap)
                state->rsp->matrix(w1, proj ? 0x03 : 0x02);   // F3D constants: PROJECTION=1, LOAD=2

            if (!proj && f5_alt_probe()) {
                const float tx = (float)rd_be_s16(ram, w1 + 2 * 12) + (float)rd_be_u16(ram, w1 + 32 + 2 * 12) / 65536.0f;
                const float ty = (float)rd_be_s16(ram, w1 + 2 * 13) + (float)rd_be_u16(ram, w1 + 32 + 2 * 13) / 65536.0f;
                const float tz = (float)rd_be_s16(ram, w1 + 2 * 14) + (float)rd_be_u16(ram, w1 + 32 + 2 * 14) / 65536.0f;
                const int16_t iy = rd_be_s16(ram, w1 + 2 * 13);
                static int n = 0; ++n;
                // Always log large/near-rail translations; otherwise sample every 90th load.
                if (iy > 8000 || iy < -8000 || (n % 90) == 0) {
                    std::fprintf(stderr, "[alt-mv] addr=%06X t=(%.1f %.1f %.1f) iy=%d m11=%.4f m22=%.4f\n",
                        w1 & 0x00FFFFFFu, tx, ty, tz, (int)iy,
                        (float)rd_be_s16(ram, w1 + 2 * 5) + (float)rd_be_u16(ram, w1 + 32 + 2 * 5) / 65536.0f,
                        (float)rd_be_s16(ram, w1 + 2 * 10) + (float)rd_be_u16(ram, w1 + 32 + 2 * 10) / 65536.0f);
                    std::fflush(stderr);
                }
            }
            if (proj && f5_proj_log()) { static int n = 0; if ((n++ % 120) == 0) {
                float M[16];
                for (int i = 0; i < 16; ++i) M[i] = (float)rd_be_s16(ram, w1 + 2 * i) + (float)rd_be_u16(ram, w1 + 32 + 2 * i) / 65536.0f;
                // Row-vector N64 layout: clip.z from M[8..10]/M[14], clip.w from M[11]/M[15].
                const float A = M[10], C = M[14], wz = M[11], ww = M[15];  // ndc.z = (z*A + w*C)/(z*wz + w*ww)
                // near: ndc.z=-1, far: ndc.z=+1 -> solve for view-space z (w=1).
                const float nearZ = (wz + ww != 0.0f) ? (C + ww) / -(A + wz) : 0.0f;   // ndc=-1
                const float farZ  = (ww - wz != 0.0f) ? (C - ww) / -(A - wz) : 0.0f;   // ndc=+1
                std::fprintf(stderr, "[f5-proj] z-row[%.4f %.4f %.4f %.4f] w-row[%.4f %.4f %.4f %.4f] near~%.1f far~%.1f\n",
                    M[8], M[9], M[10], M[11], M[12], M[13], M[14], M[15], nearZ, farZ); std::fflush(stderr); } }

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
            int32_t pmnX = 32767, pmxX = -32768, pmnY = 32767, pmxY = -32768, pmnZ = 32767, pmxZ = -32768;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t va = w1 + i * 8;
                out[i].x = (int16_t)rd_be_s16(ram, va);
                out[i].y = (int16_t)rd_be_s16(ram, va + 2);
                out[i].z = (int16_t)rd_be_s16(ram, va + 4);
                if (f5_alt_probe()) {
                    if (out[i].x < pmnX) pmnX = out[i].x; if (out[i].x > pmxX) pmxX = out[i].x;
                    if (out[i].y < pmnY) pmnY = out[i].y; if (out[i].y > pmxY) pmxY = out[i].y;
                    if (out[i].z < pmnZ) pmnZ = out[i].z; if (out[i].z > pmxZ) pmxZ = out[i].z;
                    if (i + 1 == n) {
                        static int pn = 0; ++pn;
                        const bool rail = pmnX <= -32000 || pmxX >= 32000 || pmnY <= -32000 || pmxY >= 32000 || pmnZ <= -32000 || pmxZ >= 32000;
                        if (rail || (pn % 90) == 0) {
                            const uint32_t gmCull = state->rsp->geometryModeStack[state->rsp->geometryModeStackSize - 1] & 0x3000u;
                            std::fprintf(stderr, "[alt-vtx] src=%06X n=%u x[%d..%d] y[%d..%d] z[%d..%d] cull=%04X%s\n",
                                w1 & 0x00FFFFFFu, n, pmnX, pmxX, pmnY, pmxY, pmnZ, pmxZ, gmCull, rail ? " RAIL" : "");
                            std::fflush(stderr);
                        }
                    }
                }
                out[i].flag = 0;
                out[i].s = 0;
                out[i].t = 0;
                const uint32_t c = haveColors ? rd_be_u32(ram, cbuf + i * 4) : 0xFFFFFFFFu;
                out[i].color.r = (uint8_t)(c >> 24);
                out[i].color.g = (uint8_t)(c >> 16);
                out[i].color.b = (uint8_t)(c >> 8);
                out[i].color.a = (uint8_t)c;
            }
            if (!s_cull_skip) state->rsp->setVertex(0x80000000u | F5_VTX_SCRATCH, n, 0);
            s_cache_count = n;
        }

        // 0xBD: ucode LOAD OVERLAY 0x2C = camera-facing BILLBOARD sprite -> texrect (see project memory
        // animated-effect-sprites-invisible). Hardware projects the cached center vertex and emits a
        // screen-space texrect. First-pass HLE: emit a quad around the cached center (op_04) with the
        // record color + bound texture, GPU-projected. Gated ROGUESQ_F5_SPRITES (off until validated).
        void op_bd_sprite(State* state, DisplayList** dl) {
            static bool s_on = env_on("ROGUESQ_F5_SPRITES", true);   // F5 billboard sprites (fire/smoke/explosion); ROGUESQ_F5_SPRITES=0 disables
            const uint32_t w0 = (*dl)->w0, w1 = (*dl)->w1;
            if (s_on && f5_native_active()) {
                const uint32_t slot = ((w0 >> 5) & 0x7F8u) / 0x28u;
                if (slot < s_cache_count && slot < F5_FACE_SLOT) {
                    const RSP::Vertex* cache = reinterpret_cast<const RSP::Vertex*>(state->fromRDRAM(F5_VTX_SCRATCH));
                    const RSP::Vertex c = cache[slot];
                    int16_t half = (int16_t)((*dl)[1].w1 & 0xFFFFu);   // rec[1] = (S,S) half-size
                    if (half <= 0 || half > 4000) half = 40;
                    // UV spans the actual bound tile (lrs/lrt are 10.2 fixed -> +1 texel), S10.5 into the vertex.
                    const LoadTile& T = state->rdp->tiles[0];
                    const int16_t tw = (int16_t)(((T.lrs - T.uls) >> 2) + 1) << 5;
                    const int16_t th = (int16_t)(((T.lrt - T.ult) >> 2) + 1) << 5;
                    RSP::Vertex* tmp = reinterpret_cast<RSP::Vertex*>(state->fromRDRAM(F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)));
                    const int16_t us[4] = { 0, tw, 0, tw }, vt[4] = { 0, 0, th, th };
                    const float sgx[4] = { -1, 1, -1, 1 }, sgy[4] = { -1, -1, 1, 1 };
                    // Camera-facing billboard. F5's view matrix is identity (camera is baked into the MVP),
                    // so derive the world directions that project to pure screen X/Y from the MVP rows:
                    //   screenRight_world = cross(clipY_coeffs, clipW_coeffs)
                    //   screenUp_world    = cross(clipW_coeffs, clipX_coeffs)
                    const auto& mv = state->rsp->modelViewProjMatrix;
                    const float cX[3] = { (float)mv[0][0], (float)mv[1][0], (float)mv[2][0] };  // clip.x coeffs
                    const float cY[3] = { (float)mv[0][1], (float)mv[1][1], (float)mv[2][1] };  // clip.y coeffs
                    const float cW[3] = { (float)mv[0][3], (float)mv[1][3], (float)mv[2][3] };  // clip.w coeffs
                    auto cross = [](const float a[3], const float b[3], float o[3]) {
                        o[0] = a[1]*b[2] - a[2]*b[1]; o[1] = a[2]*b[0] - a[0]*b[2]; o[2] = a[0]*b[1] - a[1]*b[0];
                    };
                    auto norm = [](float v[3]) { float l = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if (l > 1e-6f) { v[0]/=l; v[1]/=l; v[2]/=l; } };
                    float R[3], U[3]; cross(cY, cW, R); cross(cW, cX, U); norm(R); norm(U);
                    float rx = R[0], ry = R[1], rz = R[2], ux = U[0], uy = U[1], uz = U[2];
                    const bool haveView = (rx*rx + ry*ry + rz*rz) > 0.01f && (ux*ux + uy*uy + uz*uz) > 0.01f;
                    for (int k = 0; k < 4; ++k) {
                        tmp[k] = c;
                        const float dx = sgx[k] * (float)half, dy = sgy[k] * (float)half;
                        if (haveView) {
                            tmp[k].x = (int16_t)(c.x + dx * rx + dy * ux);
                            tmp[k].y = (int16_t)(c.y + dx * ry + dy * uy);
                            tmp[k].z = (int16_t)(c.z + dx * rz + dy * uz);
                        } else {
                            tmp[k].x = (int16_t)(c.x + dx);
                            tmp[k].y = (int16_t)(c.y + dy);
                        }
                        tmp[k].s = us[k]; tmp[k].t = vt[k];
                        tmp[k].color.r = tmp[k].color.g = tmp[k].color.b = tmp[k].color.a = 0xFF;  // white; PRIM carries the color
                    }
                    // Save the RDP state this sprite overrides so it does not leak into the FOLLOWING
                    // mesh/effect faces. op_bd only restored geometryMode; a leaked zSource=PRIM + primDepth
                    // (and prim color) collapses the next f5_emit_face quads to the sprite's single depth ->
                    // explosion-mesh sort artifacts. ROGUESQ_F5_SPRITE_NORESTORE=1 disables (A/B the leak).
                    static const bool s_norestore = env_on("ROGUESQ_F5_SPRITE_NORESTORE", false);
                    const uint32_t savedOMH = state->rdp->otherMode.H, savedOML = state->rdp->otherMode.L;
                    const auto savedPrimD = state->rdp->primDepthStack[state->rdp->primDepthStackSize - 1];
                    const auto savedPrimC = state->rdp->primColorStack[state->rdp->primColorStackSize - 1];
                    // Game combine FC11A7FF = color TEXEL0*PRIM, alpha TEXEL1*PRIM_a: TEXEL0 is the light
                    // RGBA16 puff, TEXEL1 the I8 shape mask, PRIM = record w1 the per-sprite tint + fade.
                    state->rdp->setPrimColor(0, 0xFF, w1);
                    // The game blender runs a FOG cycle first (otherL C4104A54); its color tracks the
                    // scene and washes the sprite. Use the plain non-fog 2-cycle alpha-over the explosion
                    // geometry uses so the sprite keeps its own color. Z_CMP on / Z_UPD off (0x10 set,
                    // 0x20 clear) + zSource=PRIM (0x04 set): depth-test the sprite without writing depth.
                    // The whole quad rasterizes at primDepth (RT64 RasterVS), so set primDepth to the
                    // center's own projected NDC depth below -- otherwise the sprite inherits a STALE
                    // primDepth from the last DL command (near on the first cinematic pass, far after the
                    // attract demo runs) which is the loop-dependent "behind terrain" bug.
                    // ROGUESQ_F5_SPRITE_NODEPTH=1 restores the no-depth-test band-aid for A/B.
                    static const bool s_nodepth = env_on("ROGUESQ_F5_SPRITE_NODEPTH", false);
                    state->rdp->setOtherMode(state->rdp->otherMode.H, s_nodepth ? 0x00504A44u : 0x00504A54u);
                    // Billboards are UNLIT (hw draws them as screen-space texrects). Force lighting off
                    // so the scene's environment light doesn't modulate the sprite (white-scene -> white,
                    // dark-scene -> black). Restore the mode after.
                    uint32_t& gm = state->rsp->geometryModeStack[state->rsp->geometryModeStackSize - 1];
                    const uint32_t savedGM = gm;
                    gm &= ~(uint32_t)G_LIGHTING;
                    state->rsp->setVertex(0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)), 4, F5_FACE_SLOT);
                    // primDepth must equal the depth a normally-projected vertex gets here. RasterVS never
                    // applies an MVP: it rebuilds clip space from the CPU's posScreen (z_clip = posScreen.z*w),
                    // and for zSource=PRIM it uses z_clip = primDepth*w. So take the depth from the verts we
                    // just transformed. Re-multiplying the center through rsp->modelViewProjMatrix (the old
                    // code) reads a STALE matrix -- it is only refreshed lazily inside setVertex -- and stamped
                    // sprites ~2x too far (RenderDoc: w=8023 quad at ndc.z 0.9996 vs true 0.9979), so lasers
                    // and particles failed the depth test against water/structures ("trails cut through").
                    {
                        const int wc = state->ext.workloadQueue->writeCursor;
                        const auto &ps = state->ext.workloadQueue->workloads[wc].drawData.posScreen;
                        float z = 0.0f;
                        for (int k = 0; k < 4; ++k) z += (float)ps[state->rsp->indices[F5_FACE_SLOT + k]][2];
                        z *= 0.25f;
                        z = z < 0.0f ? 0.0f : (z > 1.0f ? 1.0f : z);
                        state->rdp->setPrimDepth((uint16_t)(z * 32767.0f), 0);
                    }
                    state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 3, F5_FACE_SLOT + 2);
                    state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 1, F5_FACE_SLOT + 3);
                    gm = savedGM;
                    if (!s_norestore) {
                        state->rdp->setOtherMode(savedOMH, savedOML);
                        state->rdp->primDepthStack[state->rdp->primDepthStackSize - 1] = savedPrimD;
                        state->rdp->primColorStack[state->rdp->primColorStackSize - 1] = savedPrimC;
                    }
                    s_task_faces += 2;
                }
            }
            (*dl)++;  // 16-byte command: skip the extra 8 bytes like op_consume16
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
                if (idx[k] >= F5_FACE_SLOT || idx[k] >= s_cache_count) {
                    // ROGUESQ_FX_PROBE: count faces dropped by the index guard. If the animated
                    // billboards are dropped here, they never reach the draw (or the force-show path).
                    static bool s_fxr = env_on("ROGUESQ_FX_PROBE", false);
                    if (s_fxr) {
                        static uint32_t s_rej = 0, s_rejTex = 0; ++s_rej; if (st) ++s_rejTex;
                        if ((s_rej % 200) == 1 || s_rej <= 20)
                            std::fprintf(stderr, "[fx-reject] #%u (tex=%u) idx[%d]=%u cacheCount=%u n=%d tex=%06X\n",
                                s_rej, s_rejTex, k, idx[k], s_cache_count, n, state->rdp->texture.address & 0x00FFFFFFu);
                    }
                    return;   // garbage / stale index
                }
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
            // ROGUESQ_FX_PROBE: the animated explosion/smoke/fire flipbook is RGBA32 (fmt0/siz3).
            // Log distinct (tex,comb,other) tuples + a running drawn count so the real effect
            // sprites self-identify regardless of RDRAM address, and we see their exact alpha state.
            {
                static bool s_fx = env_on("ROGUESQ_FX_PROBE", false);
                if (s_fx) {
                    const LoadTile& T = state->rdp->tiles[0];
                    if (st != nullptr) {  // census ALL textured quad types (dedup by fmt/siz/comb)
                        static uint32_t s_drawn = 0; ++s_drawn;
                        const auto& prim = state->rdp->primColorStack[state->rdp->primColorStackSize - 1];
                        const auto& comb = state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1];
                        static std::unordered_set<uint64_t> s_seen;
                        uint64_t key = ((uint64_t)T.fmt << 60) ^ ((uint64_t)T.siz << 56)
                                     ^ ((uint64_t)comb.L << 4) ^ ((uint64_t)state->rdp->otherMode.L << 24);
                        if (s_seen.size() < 80 && s_seen.insert(key).second) {
                            // Source RGBA32 alpha survey: is the fire texture's alpha nonzero at
                            // its RDRAM source? If yes but sprite invisible => RT64 drops siz3 alpha.
                            const uint32_t tsrc = state->rdp->texture.address & 0x00FFFFFFu;
                            int nzA = 0, maxA = 0; uint8_t a0px[4] = {0,0,0,0};
                            for (int i = 0; i < 256; ++i) {
                                uint8_t a = state->RDRAM[(tsrc + (uint32_t)i * 4 + 3) ^ 3];
                                if (a) ++nzA; if (a > maxA) maxA = a;
                            }
                            for (int i = 0; i < 4; ++i) a0px[i] = state->RDRAM[(tsrc + i) ^ 3];
                            std::fprintf(stderr,
                                "[fx-face] #%u tex=%06X fmt=%u siz=%u texel0=(%u,%u,%u,%u) srcNZalpha=%d/256 maxA=%d n=%d cimg=%06X shade0=(%u,%u,%u,%u) haveCol=%d cbuf=%08X prim=(%.2f %.2f %.2f %.2f) combL=%08X combH=%08X otherL=%08X otherH=%08X cyc=%u filt=%u\n",
                                s_drawn, tsrc, T.fmt, T.siz, a0px[0], a0px[1], a0px[2], a0px[3], nzA, maxA, n,
                                state->rdp->colorImage.address & 0x00FFFFFFu,
                                tmp[0].color.r, tmp[0].color.g, tmp[0].color.b, tmp[0].color.a,
                                haveColors ? 1 : 0, cbuf,
                                (float)prim.x, (float)prim.y, (float)prim.z, (float)prim.w,
                                comb.L, comb.H, state->rdp->otherMode.L, state->rdp->otherMode.H,
                                state->rdp->otherMode.cycleType(), state->rdp->otherMode.textFilt());
                            std::fflush(stderr);
                        }
                    }
                }
            }
            // DIAG (ROGUESQ_LOG_GBI): the filter/cycle a MODEL face actually draws with. Point vs bilerp
            // decides whether models are blocky (point) or smooth. Bounded to the first few faces.
            { static int s_ff = 0; if (gbi_log_enabled() && ++s_ff <= 8) {
                std::fprintf(stderr, "[f5-face] n=%d textFilt=%u cycle=%u otherH=0x%08X otherL=0x%08X\n",
                    n, state->rdp->otherMode.textFilt(), state->rdp->otherMode.cycleType(),
                    state->rdp->otherMode.H, state->rdp->otherMode.L);
                std::fflush(stderr); } }
            // Effect meshes (explosion debris/exhaust, tex 0x543xxx) set zSource=PRIM (otherL 0xC810xxxx)
            // without an 0xEE, so RT64 would rasterize them at a stale primDepth. Synthesize one from the
            // face's own projected depth. Skipped when the game sent its own 0xEE (the sky pins itself to
            // 0x7FBE; overriding that drew the horizon haze over far structures). ROGUESQ_F5_MESH_PRIMDEPTH=0 A/B.
            if (!s_cull_skip) {
                state->rsp->setVertex(0x80000000u | (F5_VTX_SCRATCH + F5_FACE_SLOT * sizeof(RSP::Vertex)), n, F5_FACE_SLOT);
                if ((state->rdp->otherMode.L & 0x04u) && !s_prim_depth_from_game) {
                    static const bool s_mpd = env_on("ROGUESQ_F5_MESH_PRIMDEPTH", true);
                    if (s_mpd) {
                        // Depth from the verts just transformed (posScreen.z == the GPU depth), not a
                        // re-multiply through rsp->modelViewProjMatrix, which is stale here -- see op_bd.
                        const int wc = state->ext.workloadQueue->writeCursor;
                        const auto &ps = state->ext.workloadQueue->workloads[wc].drawData.posScreen;
                        float z = 0.0f;
                        for (int k = 0; k < n; ++k) z += (float)ps[state->rsp->indices[F5_FACE_SLOT + k]][2];
                        z /= (float)n;
                        z = z < 0.0f ? 0.0f : (z > 1.0f ? 1.0f : z);
                        state->rdp->setPrimDepth((uint16_t)(z * 32767.0f), 0);
                    }
                }
                state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 1, F5_FACE_SLOT + 2);
                if (n == 4) state->rsp->drawIndexedTri(F5_FACE_SLOT, F5_FACE_SLOT + 2, F5_FACE_SLOT + 3);
            }
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
            // Moveword 8 / 0x0A write the fog M / O words (see f5_apply_fog). F3D reads 0x0A as a light
            // color, so both are intercepted. ROGUESQ_F5_NOFOG=1 leaves fog untouched.
            { static GBIFunction s_mw = gbi->map[0xBC];
              s_mw_orig = s_mw;
              gbi->map[0xBC] = +[](State* state, DisplayList** dl) {
                  const uint32_t idx = (*dl)->w0 & 0xFF;
                  if (idx == 8 || idx == 0x0A) {
                      static bool s_nofog = env_on("ROGUESQ_F5_NOFOG", false);
                      if (s_nofog) return;
                      const double v = (double)(int32_t)(*dl)->w1 / 65536.0;
                      if (idx == 8) s_f5_fog_m = v; else s_f5_fog_o = v;
                      f5_apply_fog(state);
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
                    gbi->map[0xBD] = s_be16 ? &op_bd_sprite : &op_noop;   // sprite emit (self-gated ROGUESQ_F5_SPRITES)
                    gbi->map[0xBE] = s_be16 ? &op_consume16 : &op_noop;
                }
            }

            // RDP-state / raster wrappers (rt64_gbi_f5_rdpstate.cpp).
            gbi->map[0xFC] = &setCombine_logged;
            gbi->map[0xF5] = &setTile_logged;
            gbi->map[0xF6] = &fillRect_op02_aware;
            gbi->map[0xF7] = &setFillColor_overridden;
            gbi->map[0xB9] = [](State *state, DisplayList **dl) {
                setOtherModeL_logged(state, dl);
                if ((state->rdp->otherMode.L & 0x04u) == 0) s_prim_depth_from_game = false;
            };
            gbi->map[0xEE] = [](State *state, DisplayList **dl) {
                GBI_RDP::setPrimDepth(state, dl);
                s_prim_depth_from_game = true;
            };
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

// Called per object by a game-side hook on traverseSceneGraphRecursive's matrix-write site
// (before_vram 0x800155e4): ringDst = a0 (the matrix output cursor MEM_W[0x8011DC5C]),
// nodeId = the scene node pointer (register s2). Feeds op_01_matrix's ROGUESQ_F5_NODE_ID lookup.
// Safe no-op until that hook exists; the lookup falls back to G_EX_ID_AUTO when the map is empty.
extern "C" void rs64_f5_map_node(uint32_t ringDst, uint32_t nodeId) {
    RT64::GBI_F3DFACTOR5::f5_map_node_impl(ringDst, nodeId);
}

//
// RT64
//

#include "rt64_interpreter.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

//#define DUMP_DISPLAY_LISTS

namespace RT64 {
    static FILE *displayListFp = nullptr;

    // F5 DL-walker desync trace (ROGUESQ_DESYNC_TRACE): a ring of the last opcodes
    // dispatched, with each command's dl pointer so the byte-deltas reveal per-op
    // consumed lengths. The F5 GBI calls rt64_f5_desync_dump() when it reads a
    // garbage setColorImage (data word parsed as a command) so we can see WHICH
    // preceding op mis-advanced the cursor (the desync source).
    namespace {
        struct F5RingEntry { uint8_t op; uint32_t w0; uint32_t w1; uint64_t dlptr; };
        F5RingEntry g_f5ring[96] = {};
        uint32_t g_f5ring_pos = 0;
        int g_f5ring_on = -1;
    }
    bool rt64_f5_desync_enabled() {
        if (g_f5ring_on < 0) { const char *e = std::getenv("ROGUESQ_DESYNC_TRACE"); g_f5ring_on = (e && e[0] && e[0] != '0') ? 1 : 0; }
        return g_f5ring_on == 1;
    }
    void rt64_f5_desync_record(uint8_t op, uint32_t w0, uint32_t w1, const void *dl) {
        F5RingEntry &e = g_f5ring[g_f5ring_pos % 96];
        e.op = op; e.w0 = w0; e.w1 = w1; e.dlptr = (uint64_t)(uintptr_t)dl;
        g_f5ring_pos++;
    }
    extern "C" void rt64_f5_desync_dump(const char *why, uint32_t badW1) {
        if (!rt64_f5_desync_enabled()) return;
        static int s_dumps = 0;
        if (s_dumps >= 12) return;
        ++s_dumps;
        std::fprintf(stderr, "[desync #%d] %s badW1=0x%08X — last %d ops (op w0 w1 | bytesConsumed):\n",
                     s_dumps, why, badW1, 24);
        uint32_t start = (g_f5ring_pos >= 24) ? (g_f5ring_pos - 24) : 0;
        for (uint32_t i = start; i < g_f5ring_pos; i++) {
            const F5RingEntry &e = g_f5ring[i % 96];
            int64_t consumed = -1;
            if (i + 1 <= g_f5ring_pos - 1) { const F5RingEntry &n = g_f5ring[(i + 1) % 96]; consumed = (int64_t)(n.dlptr - e.dlptr); }
            std::fprintf(stderr, "  @%08llX op=0x%02X w0=0x%08X w1=0x%08X bytes=%lld\n", (unsigned long long)(0x80000000ull | e.dlptr), e.op, e.w0, e.w1, (long long)consumed);
        }
        std::fflush(stderr);
    }

    // Interpreter

    Interpreter::Interpreter() {
        state = nullptr;
        hleGBI = nullptr;
        extendedFunction = gbiManager.getExtendedFunction();
    }

    void Interpreter::setup(State *state) {
        this->state = state;
    }

    void Interpreter::loadUCodeGBI(uint32_t textAddress, uint32_t dataAddress, bool resetFromTask) {
        if (!resetFromTask) {
            state->flush();
        }

        const uint32_t AddressMask = 0xFFFFF8;
        const uint32_t maskedTextAddress = textAddress & AddressMask;
        const uint32_t maskedDataAddress = dataAddress & AddressMask;
        if ((UCode.textAddress != maskedTextAddress) || (UCode.dataAddress != maskedDataAddress)) {
            hleGBI = gbiManager.getGBIForUCode(state->RDRAM, maskedTextAddress, maskedDataAddress);
            if (hleGBI != nullptr) {
                state->rsp->setGBI(hleGBI);
            }

            UCode.textAddress = maskedTextAddress;
            UCode.dataAddress = maskedDataAddress;
        }

        if (hleGBI != nullptr) {
            GBIReset resetFunction = resetFromTask ? hleGBI->resetFromTask : hleGBI->resetFromLoad;
            if (resetFunction != nullptr) {
                resetFunction(state);
            }
        }
    }

    void Interpreter::processRDPLists(uint32_t dlStartAdddress, DisplayList *dlStart, DisplayList *dlEnd) {
        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        GBI *rdpGBI = state->rdp->gbi;
        constexpr unsigned int opCodeMask = 0x3F;

        // Run the command interpreter.
        assert(rdpGBI != nullptr);
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        uint32_t cmdLength;
        size_t pendingCommandRemainingBytes = state->rdp->pendingCommandRemainingBytes;

        if (dlStart >= dlEnd) {
            state->dlCpuProfiler.end();
            return;
        }

        if (pendingCommandRemainingBytes != 0) {
            // Copy the remaining command bytes from the current displaylist
            uint32_t toCopy = (uint32_t)std::min(pendingCommandRemainingBytes, (uintptr_t)dlEnd - (uintptr_t)dl);
            memcpy(state->rdp->pendingCommandBuffer.data() + state->rdp->pendingCommandCurrentBytes, dl, toCopy);

            // Modify start to skip the copied bytes
            dl = (DisplayList *)(toCopy + (uintptr_t)dl);

            // Check if we've copied all of the bytes of the command into the buffer
            if (pendingCommandRemainingBytes == toCopy) {
                // All bytes have been copied, so run the completed command
                DisplayList *pendingCommand = (DisplayList *)state->rdp->pendingCommandBuffer.data();
                opCode = (pendingCommand->w0 >> 24) & opCodeMask;
                func = rdpGBI->map[opCode];

                if (func != nullptr) {
                    func(state, &pendingCommand);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }

                state->rdp->pendingCommandCurrentBytes = 0;
                state->rdp->pendingCommandRemainingBytes = 0;
            }
            // Not all of the bytes were copied, so adjust RDP state accordingly and exit.
            else {
                state->rdp->pendingCommandCurrentBytes += toCopy;
                state->rdp->pendingCommandRemainingBytes -= toCopy;
                state->dlCpuProfiler.end();
                return;
            }
        }

        // Create a dummy pointer and pass that, since displaylist pointer incrementing is handled differently in LLE.
        DisplayList *dummy;
        while ((dl != nullptr) && ((dlEnd == nullptr) || (dl < dlEnd))) {
            opCode = (dl->w0 >> 24) & opCodeMask;

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                dummy = dl;
                extendedFunction(state, &dl);
                cmdLength = 1;
            }
            else {
                func = rdpGBI->map[opCode];
                cmdLength = state->rdp->commandWordLengths[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                // Check if this command is unfinished and store the partial contents if so.
                if (dl + cmdLength > dlEnd) {
                    uint32_t toCopy = (uint32_t)((uintptr_t)dlEnd - (uintptr_t)dl);
                    memcpy(state->rdp->pendingCommandBuffer.data(), dl, toCopy);
                    state->rdp->pendingCommandCurrentBytes = toCopy;
                    state->rdp->pendingCommandRemainingBytes = cmdLength * sizeof(DisplayList) - toCopy;
                    break;
                }

                if (func != nullptr) {
                    dummy = dl;
                    func(state, &dummy);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }
            }

            if (dl != nullptr) {
                dl += cmdLength;
            }
        }

        state->dlCpuProfiler.end();
    }

    void Interpreter::processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart) {
        assert(hleGBI != nullptr);

        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        // Run the command interpreter.
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        const bool desyncTrace = rt64_f5_desync_enabled();
        while (dl != nullptr) {
            opCode = (dl->w0 >> 24);

            if (desyncTrace) rt64_f5_desync_record(opCode, dl->w0, dl->w1, reinterpret_cast<const void *>(reinterpret_cast<const uint8_t *>(dl) - state->RDRAM));   // RDRAM offset, not host pointer

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                extendedFunction(state, &dl);
            }
            else {
                func = hleGBI->map[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                if (func != nullptr) {
                    func(state, &dl);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown opCode (GBI %u): %u / 0x%X", uint32_t(hleGBI->ucode), opCode, opCode);
                }
            }

            if (dl != nullptr) {
                dl++;
            }
        }

        state->dlCpuProfiler.end();
    }
};

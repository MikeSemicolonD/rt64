//
// RT64
//

#pragma once

#include <cstdint>
#include <string>

#include <json/json.hpp>

namespace RT64 {
    struct State;

    // RS64: MCP bridge requests that need RT64 state, run on the gfx thread by drain() at the top of State::updateScreen.
    namespace rs64mcp {
        bool has(const std::string &method);
        void post(uint32_t tag, const nlohmann::json &id, const std::string &method, const nlohmann::json &params);
        bool takeReply(uint32_t &tag, nlohmann::json &reply);
        void drain(State &state);

        // Host-side pause control: applied by the next drain; drainCount() advances once per updateScreen.
        void requestPause(bool on);
        // Turns off the debug views the MCP tools set (framebuffer view, draw cut-off, free camera, texture dumper) at the next drain.
        void requestReset();
        bool paused();
        uint64_t drainCount();
    };
};

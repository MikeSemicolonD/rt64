//
// RT64
//

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

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

        // "Drawn" watch for pause_when_drawn: '|'-separated mesh / sprite key prefixes (r_pow, spr:explo_tex). The GBI reports each key that puts a
        // vertex inside the view (drawWatchSeen); the first match is kept until disarmed. Inline so builds without the bridge compile it (never armed there).
        struct DrawWatch {
            std::atomic<bool> armed{ false };
            std::mutex mutex;
            std::vector<std::string> prefixes;
            std::string hitKey;
        };

        inline DrawWatch &drawWatch() {
            static DrawWatch w;
            return w;
        }

        inline void armDrawWatch(const std::string &prefixes) {
            DrawWatch &w = drawWatch();
            std::lock_guard<std::mutex> lock(w.mutex);
            w.prefixes.clear();
            w.hitKey.clear();
            size_t start = 0;
            while (start <= prefixes.size()) {
                const size_t bar = prefixes.find('|', start);
                const std::string p = prefixes.substr(start, (bar == std::string::npos) ? std::string::npos : bar - start);
                if (!p.empty()) {
                    w.prefixes.push_back(p);
                }
                if (bar == std::string::npos) {
                    break;
                }
                start = bar + 1;
            }
            w.armed = !w.prefixes.empty();
        }

        inline void disarmDrawWatch() {
            DrawWatch &w = drawWatch();
            std::lock_guard<std::mutex> lock(w.mutex);
            w.armed = false;
            w.prefixes.clear();
            w.hitKey.clear();
        }

        inline bool drawWatchArmed() {
            return drawWatch().armed.load(std::memory_order_relaxed);
        }

        inline void drawWatchSeen(const std::string &key) {
            DrawWatch &w = drawWatch();
            std::lock_guard<std::mutex> lock(w.mutex);
            if (!w.armed || !w.hitKey.empty()) {
                return;
            }
            for (const std::string &p : w.prefixes) {
                if (key.compare(0, p.size(), p) == 0) {
                    w.hitKey = key;
                    return;
                }
            }
        }

        inline bool drawWatchHit(std::string &key) {
            DrawWatch &w = drawWatch();
            std::lock_guard<std::mutex> lock(w.mutex);
            key = w.hitKey;
            return !w.hitKey.empty();
        }
    };
};

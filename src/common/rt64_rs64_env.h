#pragma once

// ROGUESQ_* knobs with the host's recomp::dbg::env_* semantics: unset or empty gives the default, and on() treats any value not starting with '0' as on.
// Read them into a static const at hot call sites.

#include <cstdlib>

namespace rs64env {
    inline const char *str(const char *name) {
        const char *v = std::getenv(name);
        return ((v != nullptr) && (*v != '\0')) ? v : nullptr;
    }

    inline bool on(const char *name, bool def = false) {
        const char *v = str(name);
        return (v != nullptr) ? (*v != '0') : def;
    }

    inline int integer(const char *name, int def = 0) {
        const char *v = str(name);
        return (v != nullptr) ? std::atoi(v) : def;
    }

    inline float number(const char *name, float def = 0.0f) {
        const char *v = str(name);
        return (v != nullptr) ? float(std::atof(v)) : def;
    }

    inline unsigned u32(const char *name, unsigned def = 0u) {
        const char *v = str(name);
        return (v != nullptr) ? unsigned(std::strtoul(v, nullptr, 0)) : def;
    }
}

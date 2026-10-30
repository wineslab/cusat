/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trx/radio_loader.h"

#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

std::unique_ptr<radio_device> load_radio_device(const std::string& driver)
{
    // Search order: explicit env override, then the bare soname (resolved
    // via LD_LIBRARY_PATH / ld cache / system paths), then a copy located
    // next to this executable (the in-tree build layout).
    const std::string soname = "libradio_" + driver + ".so";
    std::vector<std::string> candidates;
    if (const char* env = getenv("CUSAT_RADIO_PLUGIN"); env && *env)
        candidates.emplace_back(env);
    candidates.emplace_back(soname);
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = '\0';
        std::string d(exe);
        auto slash = d.find_last_of('/');
        if (slash != std::string::npos)
            candidates.push_back(d.substr(0, slash + 1) + soname);
    }

    void* handle = nullptr;
    std::string used;
    std::string last_err;
    for (const auto& p : candidates) {
        /* RTLD_LOCAL (default): keep the plugin's (GPL) symbols, incl. UHD,
         * private to the plugin. RTLD_NOW: fail fast on a broken plugin. */
        handle = dlopen(p.c_str(), RTLD_NOW);
        if (handle) { used = p; break; }
        const char* e = dlerror();
        last_err = e ? e : "(unknown)";
    }
    if (!handle) {
        fprintf(stderr, "load_radio_device: could not load radio plugin "
                "(tried %zu paths; last error: %s)\n",
                candidates.size(), last_err.c_str());
        return nullptr;
    }

    dlerror();  // clear
    auto create = reinterpret_cast<radio_device_create_fn>(
        dlsym(handle, "radio_device_create"));
    const char* sym_err = dlerror();
    if (sym_err || !create) {
        fprintf(stderr, "load_radio_device: dlsym(radio_device_create) "
                "failed in %s: %s\n", used.c_str(),
                sym_err ? sym_err : "(null symbol)");
        dlclose(handle);
        return nullptr;
    }

    radio_device* dev = create();
    if (!dev) {
        fprintf(stderr, "load_radio_device: radio_device_create() "
                "returned null\n");
        dlclose(handle);
        return nullptr;
    }

    fprintf(stderr, "load_radio_device: loaded radio plugin \"%s\"\n",
            used.c_str());
    /* Retain the handle for process lifetime — the plugin must stay mapped
     * while the device is in use. */
    return std::unique_ptr<radio_device>(dev);
}

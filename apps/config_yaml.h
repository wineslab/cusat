/*
 * Minimal YAML config parser for dvbs2-uhd-trx.
 *
 * Supports: sections ("key:"), key-value pairs ("key: value"),
 * inline lists ("[1, 2, 3]"), comments (#), and quoted strings.
 * One level of nesting only (section.key).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CONFIG_YAML_H
#define CONFIG_YAML_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct yaml_config {
    // Flat storage: "section.key" -> "value"
    std::map<std::string, std::string> entries;

    bool load(const char* path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "config: cannot open %s\n", path);
            return false;
        }
        std::string section;
        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            lineno++;
            // Strip comment (outside quotes)
            size_t hash = std::string::npos;
            bool in_quote = false;
            for (size_t i = 0; i < line.size(); i++) {
                if (line[i] == '"' || line[i] == '\'') in_quote = !in_quote;
                if (line[i] == '#' && !in_quote) { hash = i; break; }
            }
            if (hash != std::string::npos) line = line.substr(0, hash);

            // Trim
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);
            if (line.empty()) continue;

            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;

            std::string key = line.substr(0, colon);
            std::string val = (colon + 1 < line.size()) ? line.substr(colon + 1) : "";

            // Trim key and value
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t");
                size_t b = s.find_last_not_of(" \t");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(key);
            trim(val);

            // Strip surrounding quotes from value
            if (val.size() >= 2 &&
                ((val.front() == '"' && val.back() == '"') ||
                 (val.front() == '\'' && val.back() == '\'')))
                val = val.substr(1, val.size() - 2);

            if (val.empty()) {
                // Section header
                section = key;
            } else {
                // Key-value pair
                std::string full_key = section.empty() ? key : section + "." + key;
                entries[full_key] = val;
            }
        }
        fprintf(stderr, "config: loaded %s (%zu entries)\n", path, entries.size());
        return true;
    }

    bool has(const std::string& key) const {
        return entries.count(key) > 0;
    }

    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = entries.find(key);
        return (it != entries.end()) ? it->second : def;
    }

    double get_double(const std::string& key, double def = 0.0) const {
        auto it = entries.find(key);
        return (it != entries.end()) ? std::stod(it->second) : def;
    }

    int get_int(const std::string& key, int def = 0) const {
        auto it = entries.find(key);
        return (it != entries.end()) ? std::atoi(it->second.c_str()) : def;
    }

    float get_float(const std::string& key, float def = 0.0f) const {
        auto it = entries.find(key);
        return (it != entries.end()) ? std::stof(it->second) : def;
    }

    bool get_bool(const std::string& key, bool def = false) const {
        auto it = entries.find(key);
        if (it == entries.end()) return def;
        const std::string& v = it->second;
        return (v == "true" || v == "yes" || v == "1" || v == "on");
    }

    std::set<int> get_int_list(const std::string& key) const {
        std::set<int> result;
        auto it = entries.find(key);
        if (it == entries.end()) return result;
        std::string v = it->second;
        // Strip [ ]
        if (!v.empty() && v.front() == '[') v = v.substr(1);
        if (!v.empty() && v.back() == ']') v.pop_back();
        std::istringstream ss(v);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t a = tok.find_first_not_of(" \t");
            if (a != std::string::npos)
                result.insert(std::atoi(tok.substr(a).c_str()));
        }
        return result;
    }

    // Ordered variant of get_int_list — order matters for e.g. usrpproxy
    // tx/rx channel lists (OAI logical channel k -> physical channels[k]).
    std::vector<int> get_int_vector(const std::string& key) const {
        std::vector<int> result;
        auto it = entries.find(key);
        if (it == entries.end()) return result;
        std::string v = it->second;
        if (!v.empty() && v.front() == '[') v = v.substr(1);
        if (!v.empty() && v.back() == ']') v.pop_back();
        std::istringstream ss(v);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t a = tok.find_first_not_of(" \t");
            if (a != std::string::npos)
                result.push_back(std::atoi(tok.substr(a).c_str()));
        }
        return result;
    }
};

#endif /* CONFIG_YAML_H */

// SPDX-License-Identifier: Apache-2.0
#include "farrow_interpolator.h"
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

using namespace cusat_dsp;

static void usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s --in-file <iq> --out-file <iq> --in-sps <sps> --out-sps <sps>\n",
            prog);
}

static bool write_all(FILE* f, const void* data, size_t bytes)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    while (bytes > 0) {
        size_t n = fwrite(p, 1, bytes, f);
        if (n == 0)
            return false;
        p += n;
        bytes -= n;
    }
    return true;
}

int main(int argc, char** argv)
{
    std::string in_file, out_file;
    double in_sps = 0.0, out_sps = 0.0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--in-file" && i + 1 < argc) in_file = argv[++i];
        else if (arg == "--out-file" && i + 1 < argc) out_file = argv[++i];
        else if (arg == "--in-sps" && i + 1 < argc) in_sps = atof(argv[++i]);
        else if (arg == "--out-sps" && i + 1 < argc) out_sps = atof(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }
    if (in_file.empty() || out_file.empty() || in_sps <= 0.0 || out_sps <= 0.0) {
        usage(argv[0]);
        return 1;
    }

    FILE* in = fopen(in_file.c_str(), "rb");
    if (!in) {
        fprintf(stderr, "iq-farrow-resample: cannot open %s: %s\n",
                in_file.c_str(), strerror(errno));
        return 1;
    }
    FILE* out = fopen(out_file.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "iq-farrow-resample: cannot open %s: %s\n",
                out_file.c_str(), strerror(errno));
        fclose(in);
        return 1;
    }

    const double stride = in_sps / out_sps;
    farrow_interpolator resampler(stride);
    constexpr int CHUNK = 1 << 20;
    std::vector<complexf> input(CHUNK);
    std::vector<complexf> output((size_t)(CHUNK / stride + 4096));
    uint64_t total_in = 0, total_out = 0;

    while (true) {
        size_t n_read = fread(input.data(), sizeof(complexf), input.size(), in);
        if (n_read == 0)
            break;
        int consumed = 0;
        int n_out = resampler.resample(input.data(), (int)n_read,
                                       output.data(), (int)output.size(),
                                       consumed);
        if (consumed != (int)n_read) {
            fprintf(stderr,
                    "iq-farrow-resample: output buffer clipped (%d/%zu consumed)\n",
                    consumed, n_read);
            fclose(in);
            fclose(out);
            return 1;
        }
        if (!write_all(out, output.data(), (size_t)n_out * sizeof(complexf))) {
            fprintf(stderr, "iq-farrow-resample: write failed: %s\n", strerror(errno));
            fclose(in);
            fclose(out);
            return 1;
        }
        total_in += n_read;
        total_out += (uint64_t)n_out;
    }

    fclose(in);
    fclose(out);
    fprintf(stderr, "iq-farrow-resample: %.6f -> %.6f sps, stride %.9f, %llu -> %llu samples\n",
            in_sps, out_sps, stride,
            (unsigned long long)total_in,
            (unsigned long long)total_out);
    return 0;
}

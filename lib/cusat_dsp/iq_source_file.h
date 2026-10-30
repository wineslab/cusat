/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File/pipe IQ source: reads interleaved float32 I/Q samples from a file
 * descriptor (stdin, named pipe, or regular file).
 */

#ifndef INCLUDED_CUSAT_DSP_IQ_SOURCE_FILE_H
#define INCLUDED_CUSAT_DSP_IQ_SOURCE_FILE_H

#include "iq_source.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>

namespace cusat_dsp {

class iq_source_file : public iq_source
{
public:
    /**
     * @brief Construct from file descriptor (caller manages lifetime).
     */
    explicit iq_source_file(int fd) : d_fd(fd), d_owns_fd(false) {}

    /**
     * @brief Construct from file path (opens the file).
     */
    explicit iq_source_file(const std::string& path) : d_owns_fd(true)
    {
        d_fd = open(path.c_str(), O_RDONLY);
        if (d_fd < 0) {
            throw std::runtime_error("Cannot open IQ file: " + path);
        }
    }

    ~iq_source_file() override
    {
        if (d_owns_fd && d_fd >= 0)
            close(d_fd);
    }

    int read(complexf* buf, int max_samples) override
    {
        // Each complex sample is 2 floats (8 bytes)
        const size_t bytes_wanted = max_samples * sizeof(complexf);
        size_t total_read = 0;
        uint8_t* dst = reinterpret_cast<uint8_t*>(buf);

        while (total_read < bytes_wanted) {
            ssize_t n = ::read(d_fd, dst + total_read, bytes_wanted - total_read);
            if (n <= 0)
                break; // EOF or error
            total_read += n;
        }

        return (int)(total_read / sizeof(complexf));
    }

private:
    int d_fd;
    bool d_owns_fd;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_IQ_SOURCE_FILE_H */

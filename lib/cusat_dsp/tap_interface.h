/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TAP network interface for GSE transport (Linux only).
 */

#ifndef INCLUDED_CUSAT_DSP_TAP_INTERFACE_H
#define INCLUDED_CUSAT_DSP_TAP_INTERFACE_H

#include <chrono>
#include <cstdint>
#include <string>

namespace cusat_dsp {

class tap_interface
{
public:
    /**
     * @brief Open or attach to a TAP device.
     *
     * On Linux, opens /dev/net/tun and creates/attaches to the named TAP device
     * with IFF_TAP | IFF_NO_PI (no packet info header).
     *
     * @param name TAP device name (e.g. "dvbs2tap0")
     * @param mtu MTU (informational, actual MTU is set via ip link)
     * @throws std::runtime_error if TAP not supported or creation fails
     */
    tap_interface(const std::string& name, int mtu = 1500);
    ~tap_interface();

    /** Read one Ethernet frame (blocking). Returns frame length, or -1 on error. */
    int read_frame(uint8_t* buf, int max_len);

    /** Write one Ethernet frame (blocking). Returns bytes written, or -1 on error. */
    int write_frame(const uint8_t* buf, int len);

    /** Write one Ethernet frame (non-blocking). Returns 0 if would block (dropped). */
    int write_frame_nonblocking(const uint8_t* buf, int len);

    /** Set the TAP fd to non-blocking mode. */
    void set_nonblocking(bool enable);

    int fd() const { return d_fd; }
    const std::string& name() const { return d_name; }

    /**
     * Get the TAP interface's current hardware (MAC) address (6 bytes).
     *
     * Network managers may assign a persistent/random MAC shortly after the
     * TAP is created.  Refresh periodically so reconstructed GSE Ethernet
     * frames continue to target the address that the kernel currently owns.
     */
    const uint8_t* mac();

private:
    int d_fd;
    int d_control_fd;
    std::string d_name;
    uint8_t d_mac[6] = {};
    std::chrono::steady_clock::time_point d_next_mac_refresh{};
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_TAP_INTERFACE_H */

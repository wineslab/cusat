/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tap_interface.h"
#include <cstring>
#include <stdexcept>
#include <unistd.h>

#ifdef __linux__
#include <fcntl.h>
#include <linux/if_tun.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <errno.h>
#endif

namespace cusat_dsp {

tap_interface::tap_interface(const std::string& name, int mtu)
    : d_fd(-1), d_control_fd(-1), d_name(name)
{
#ifdef __linux__
    (void)mtu;
    d_fd = open("/dev/net/tun", O_RDWR);
    if (d_fd < 0)
        throw std::runtime_error("Cannot open /dev/net/tun: " + std::string(strerror(errno)));

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    ifr.ifr_flags = IFF_TAP | IFF_NO_PI;
    strncpy(ifr.ifr_name, name.c_str(), IFNAMSIZ - 1);

    if (ioctl(d_fd, TUNSETIFF, &ifr) < 0) {
        close(d_fd);
        d_fd = -1;
        throw std::runtime_error("Cannot create TAP device '" + name +
                                 "': " + std::string(strerror(errno)));
    }
    d_name = ifr.ifr_name;

    // Read TAP interface MAC address (needs a regular socket, not the tun fd)
    d_control_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (d_control_fd >= 0) {
        struct ifreq ifr_mac;
        memset(&ifr_mac, 0, sizeof(ifr_mac));
        strncpy(ifr_mac.ifr_name, d_name.c_str(), IFNAMSIZ - 1);
        if (ioctl(d_control_fd, SIOCGIFHWADDR, &ifr_mac) == 0)
            memcpy(d_mac, ifr_mac.ifr_hwaddr.sa_data, 6);
    }
#else
    (void)mtu;
    throw std::runtime_error(
        "TAP interface not supported on this platform. Use --pipe mode or run on Linux.");
#endif
}

tap_interface::~tap_interface()
{
    if (d_fd >= 0)
        close(d_fd);
    if (d_control_fd >= 0)
        close(d_control_fd);
}

const uint8_t* tap_interface::mac()
{
#ifdef __linux__
    const auto now = std::chrono::steady_clock::now();
    if (d_control_fd >= 0 && now >= d_next_mac_refresh) {
        struct ifreq ifr_mac;
        memset(&ifr_mac, 0, sizeof(ifr_mac));
        strncpy(ifr_mac.ifr_name, d_name.c_str(), IFNAMSIZ - 1);
        if (ioctl(d_control_fd, SIOCGIFHWADDR, &ifr_mac) == 0)
            memcpy(d_mac, ifr_mac.ifr_hwaddr.sa_data, 6);
        d_next_mac_refresh = now + std::chrono::milliseconds(250);
    }
#endif
    return d_mac;
}

int tap_interface::read_frame(uint8_t* buf, int max_len)
{
    return static_cast<int>(read(d_fd, buf, max_len));
}

int tap_interface::write_frame(const uint8_t* buf, int len)
{
    return static_cast<int>(write(d_fd, buf, len));
}

void tap_interface::set_nonblocking(bool enable)
{
#ifdef __linux__
    int flags = fcntl(d_fd, F_GETFL, 0);
    if (enable)
        fcntl(d_fd, F_SETFL, flags | O_NONBLOCK);
    else
        fcntl(d_fd, F_SETFL, flags & ~O_NONBLOCK);
#else
    (void)enable;
#endif
}

int tap_interface::write_frame_nonblocking(const uint8_t* buf, int len)
{
    ssize_t n = write(d_fd, buf, len);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return 0; // dropped
    return static_cast<int>(n);
}

} // namespace cusat_dsp

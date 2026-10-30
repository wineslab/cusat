/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-agnostic sample-streaming radio interface for dvbs2-uhd-trx.
 *
 * The concrete driver is supplied by a runtime-loaded plugin
 * (libradio_uhd.so) so the GPLv3 USRP Hardware Driver never links into this
 * (Apache-2.0) application or cusat_dsp. See trx/radio_loader.h.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/** Radio bring-up parameters (UHD-free mirror of the relevant trx_options). */
struct radio_config {
    std::string usrp_args;
    bool demo_control_enabled = false;
    std::string clock_source = "internal";
    bool   do_tx = false, do_rx = false;
    double rate = 0.0;
    // Symbol-rate TX fronthaul (tx_symbols): the TX stream leaves the host at
    // this rate (= symbol rate, pre-RRC) while `rate` remains the RX sample
    // rate. 0 = TX and RX share `rate` (normal operation).
    double tx_rate = 0.0;
    // RX functional-split rung 3: ask the RU (rrc_rx mode 2, feedforward
    // energy-search sync) to recover symbol timing and ship one on-time
    // symbol per symbol; `rate` is then the symbol rate (1 sps at the host).
    bool   rx_symbol_sync = false;
    double tx_freq = 0.0, rx_freq = 0.0;
    double tx_gain = 0.0, rx_gain = 0.0;
    int    tx_channel = 0, rx_channel = 0;
    int    uhd_num_send_frames = 16;
    int    uhd_send_buff_size = 0;
    int    uhd_send_frame_size = 0;
    int    uhd_recv_frame_size = 0;

    // UHD subdev spec applied to TX and RX right after multi_usrp::make()
    // (empty = UHD default). Needed when DVB uses channels outside the
    // default (e.g. 1 and 3) and/or the usrpproxy tenant uses channel 0.
    std::string subdev_spec;

    // DIFI driver (libradio_difi.so) endpoints: IQ-over-UDP per
    // IEEE-ISTO 4900. Ignored by the UHD driver.

    // Embedded usrpproxy tenant server — exposes a subset of the USRP's
    // physical channels to an external OAI softmodem over a POSIX shm ring.
    // Handled entirely inside the UHD plugin (shares the multi_usrp); the
    // Apache-2.0 app only forwards these values.
    bool     usrpproxy_enabled = false;
    std::string usrpproxy_shm_name = "/usrpproxy";
    std::string usrpproxy_tx_antenna;
    std::string usrpproxy_rx_antenna;
    std::vector<int> usrpproxy_tx_channels;
    std::vector<int> usrpproxy_rx_channels;
    int      usrpproxy_tx_core = -1;
    int      usrpproxy_rx_core = -1;
    uint64_t usrpproxy_tx_ring_size = 16ull * 1024 * 1024;
    uint64_t usrpproxy_rx_ring_size = 16ull * 1024 * 1024;
};

enum class radio_rx_error { none, timeout, overflow, other };

struct radio_rx_meta {
    radio_rx_error error = radio_rx_error::none;
};

struct radio_tx_meta {
    bool   start_of_burst = false;
    bool   end_of_burst   = false;
    bool   has_time_spec  = false;
    double time_spec      = 0.0;   /* device time, seconds */
};

/** One async TX status message (e.g. an underflow notification). */
struct radio_async_msg {
    bool     present        = false; /* a message was dequeued */
    bool     underflow      = false; /* device ran out of TX samples */
    bool     sequence_error = false; /* host-to-device packet loss */
    bool     time_error     = false; /* timed command arrived late */
    unsigned event_code     = 0;     /* raw driver event code */
    double   time_spec      = 0.0;   /* device timestamp, seconds */
};

/**
 * Abstract sample-streaming radio. Sample buffers are interleaved complex
 * float (fc32) in host memory. The concrete driver is provided by a plugin.
 */
struct radio_live_stats {
    uint64_t rx_wire_bytes = 0, tx_wire_bytes = 0, rx_packets = 0, switches = 0, clips = 0;
    int mode = 0, shift = 0, dbits = 0;
};

class radio_device {
public:
    virtual ~radio_device() = default;
    virtual radio_live_stats live_stats() const { return {}; }

    /** Create the device, tune, sync time to PPS, create streamers.
     *  Prints the historical banner lines to stderr. False on failure. */
    virtual bool   setup(const radio_config& cfg) = 0;

    /** Current device time, seconds. */
    virtual double get_time_now() = 0;

    /** RX functional-split rung 3: shift the RU timing estimator between
     *  acquisition and tracking gear (DU-driven coherence gating). Locked =
     *  slow gear: much rarer estimates (same per-estimate gains) so the
     *  estimator's phase-step jitter nearly vanishes while the loop still
     *  tracks real clock drift — a full freeze would dead-reckon the noise
     *  residual in the rate integrator into hundreds of samples/s of drift.
     *  No-op for drivers without RU sync. */
    virtual void rx_sync_gear(bool /*locked*/) {}

    virtual size_t rx_max_samps() = 0;
    virtual size_t tx_max_samps() = 0;

    /** Start continuous RX, gated to begin at device time at_time_secs. */
    virtual void   start_rx_continuous(double at_time_secs) = 0;
    virtual void   stop_rx() = 0;

    /** Receive up to nsamps fc32 samples; returns count, sets md.error. */
    virtual size_t recv(void* buf, size_t nsamps,
                        radio_rx_meta& md, double timeout) = 0;

    /** Send nsamps fc32 samples with the given burst metadata; returns the
     *  count actually sent. buf may be null when nsamps==0 (end-of-burst). */
    virtual size_t send(const void* buf, size_t nsamps,
                        const radio_tx_meta& md, double timeout) = 0;

    /** Poll (non-blocking) one TX async status message. */
    virtual radio_async_msg poll_tx_async() = 0;
};

/* Entry point exported by the radio plugin (GPL); resolved via dlsym. */
extern "C" radio_device* radio_device_create();
using radio_device_create_fn = radio_device* (*)();

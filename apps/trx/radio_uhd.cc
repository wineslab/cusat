/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * UHD radio plugin for dvbs2-uhd-trx.
 *
 * This source file is Apache-2.0. It links the GPLv3 USRP Hardware Driver
 * (UHD), so a distributed libradio_uhd.so binary is governed by GPLv3 as a
 * combined work (Apache-2.0 is one-way GPLv3-compatible). The plugin is a
 * standalone shared object loaded at runtime via dlopen (see
 * trx/radio_loader), implementing the hardware-agnostic radio_device
 * interface declared in trx/radio_device.h — so UHD never links into the
 * application or cusat_dsp; the same isolation model OpenAirInterface uses
 * for its USRP device plugin. radio_difi.cc follows the same pattern.
 */

#include "trx/radio_device.h"

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/usrp/subdev_spec.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/types/time_spec.hpp>
#include <uhd/types/metadata.hpp>
#include <uhd/stream.hpp>

#include "tenant_server.h"   // from usrpproxy_mvp (PUBLIC include dir)

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

class uhd_radio_device : public radio_device {
public:
    // Stop the embedded tenant server (if any) before the multi_usrp it
    // borrows is released. Member order below also guarantees this, but be
    // explicit since the tenant holds streamers on the shared handle.
    ~uhd_radio_device() override
    {
        if (d_tenant) d_tenant->stop();
        d_tenant.reset();
    }

    bool   setup(const radio_config& cfg) override;
    double get_time_now() override
    {
        return d_usrp->get_time_now().get_real_secs();
    }
    size_t rx_max_samps() override
    {
        return d_rx ? d_rx->get_max_num_samps() : 0;
    }
    size_t tx_max_samps() override
    {
        return d_tx ? d_tx->get_max_num_samps() : 0;
    }
    void   start_rx_continuous(double at_time_secs) override;
    void   stop_rx() override;
    size_t recv(void* buf, size_t nsamps, radio_rx_meta& md,
                double timeout) override;
    size_t send(const void* buf, size_t nsamps, const radio_tx_meta& md,
                double timeout) override;
    radio_async_msg poll_tx_async() override;

private:
    uhd::usrp::multi_usrp::sptr d_usrp;
    uhd::tx_streamer::sptr      d_tx;
    uhd::rx_streamer::sptr      d_rx;
    uhd::rx_metadata_t          d_rx_md;
    // Declared last so it is destroyed first (before d_usrp).
    std::unique_ptr<usrpproxy::TenantServer> d_tenant;
};

bool uhd_radio_device::setup(const radio_config& opt)
{
    std::string device_args = opt.usrp_args;
    auto add_device_arg = [&](const char* key, int value) {
        if (value <= 0) return;
        if (!device_args.empty()) device_args += ',';
        device_args += key;
        device_args += '=';
        device_args += std::to_string(value);
    };
    add_device_arg("send_frame_size", opt.uhd_send_frame_size);
    add_device_arg("recv_frame_size", opt.uhd_recv_frame_size);

    fprintf(stderr, "radio_uhd: creating USRP with args: %s\n",
            device_args.c_str());
    d_usrp = uhd::usrp::multi_usrp::make(device_args);
    if (!d_usrp) {
        fprintf(stderr, "radio_uhd: multi_usrp::make returned null\n");
        return false;
    }

    // Apply subdev spec before any set_*_rate / get_*_stream call. Required
    // whenever DVB (channels 1/3) and/or the usrpproxy tenant (channel 0)
    // address UHD channels outside the default (usually A:0 B:0).
    if (!opt.subdev_spec.empty()) {
        fprintf(stderr, "radio_uhd: subdev_spec=\"%s\"\n", opt.subdev_spec.c_str());
        d_usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t(opt.subdev_spec));
        d_usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t(opt.subdev_spec));
    }

    // Keep UHD's native model (ordered subdevices + logical channel indices),
    // but make the physical allocation explicit in the startup log and reject
    // invalid/conflicting selections before any tune or streamer call.
    std::vector<std::string> subdevs;
    if (!opt.subdev_spec.empty()) {
        std::istringstream is(opt.subdev_spec);
        for (std::string token; is >> token;) subdevs.push_back(token);
        for (size_t i = 0; i < subdevs.size(); ++i)
            fprintf(stderr, "radio_uhd: logical channel %zu -> %s\n",
                    i, subdevs[i].c_str());
    }
    const size_t ntx = d_usrp->get_tx_num_channels();
    const size_t nrx = d_usrp->get_rx_num_channels();
    auto valid_channel = [](int ch, size_t count) {
        return ch >= 0 && static_cast<size_t>(ch) < count;
    };
    if ((opt.do_tx && !valid_channel(opt.tx_channel, ntx)) ||
        (opt.do_rx && !valid_channel(opt.rx_channel, nrx))) {
        fprintf(stderr,
                "radio_uhd: ERROR: selected DVB channels TX=%d RX=%d, "
                "but UHD exposes %zu TX / %zu RX channels\n",
                opt.tx_channel, opt.rx_channel, ntx, nrx);
        return false;
    }
    auto print_selection = [&](const char* owner, const char* direction, int ch) {
        if (!valid_channel(ch, direction[0] == 'T' ? ntx : nrx)) return;
        if (static_cast<size_t>(ch) < subdevs.size())
            fprintf(stderr, "radio_uhd: %s %s channel %d -> %s\n",
                    owner, direction, ch, subdevs[ch].c_str());
        else
            fprintf(stderr, "radio_uhd: %s %s channel %d\n",
                    owner, direction, ch);
    };
    if (opt.do_tx) print_selection("cuSAT", "TX", opt.tx_channel);
    if (opt.do_rx) print_selection("cuSAT", "RX", opt.rx_channel);

    auto validate_proxy = [&](const char* direction,
                              const std::vector<int>& channels,
                              size_t count, int dvb_channel, bool dvb_active) {
        std::vector<int> seen;
        for (int ch : channels) {
            if (!valid_channel(ch, count)) {
                fprintf(stderr,
                        "radio_uhd: ERROR: usrpproxy %s channel %d is outside "
                        "the %zu exposed channels\n", direction, ch, count);
                return false;
            }
            if (std::find(seen.begin(), seen.end(), ch) != seen.end() ||
                (dvb_active && ch == dvb_channel)) {
                fprintf(stderr,
                        "radio_uhd: ERROR: conflicting %s allocation on "
                        "logical channel %d\n", direction, ch);
                return false;
            }
            seen.push_back(ch);
            print_selection("usrpproxy", direction, ch);
        }
        return true;
    };
    if (opt.usrpproxy_enabled &&
        (!validate_proxy("TX", opt.usrpproxy_tx_channels, ntx,
                         opt.tx_channel, opt.do_tx) ||
         !validate_proxy("RX", opt.usrpproxy_rx_channels, nrx,
                         opt.rx_channel, opt.do_rx)))
        return false;

    if (opt.clock_source != "internal") {
        fprintf(stderr, "radio_uhd: clock_source=%s\n",
                opt.clock_source.c_str());
        d_usrp->set_clock_source(opt.clock_source);
        d_usrp->set_time_source(opt.clock_source);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        try {
            if (d_usrp->get_mboard_sensor("ref_locked").to_bool())
                fprintf(stderr, "radio_uhd: reference clock locked OK\n");
            else
                fprintf(stderr, "radio_uhd: WARNING — reference clock NOT locked!\n");
        } catch (...) {
            fprintf(stderr, "radio_uhd: could not read ref_locked sensor\n");
        }
    }

    if (opt.do_tx) {
        d_usrp->set_tx_rate(opt.rate, opt.tx_channel);
        d_usrp->set_tx_freq(uhd::tune_request_t(opt.tx_freq), opt.tx_channel);
        d_usrp->set_tx_gain(opt.tx_gain, opt.tx_channel);
        d_usrp->set_tx_antenna("TX/RX", opt.tx_channel);
        fprintf(stderr, "radio_uhd: TX ch%d freq=%.3f MHz gain=%.1f dB rate=%.3f Msps\n",
                opt.tx_channel, opt.tx_freq / 1e6, opt.tx_gain, opt.rate / 1e6);
    }

    if (opt.do_rx) {
        d_usrp->set_rx_rate(opt.rate, opt.rx_channel);
        d_usrp->set_rx_freq(uhd::tune_request_t(opt.rx_freq), opt.rx_channel);
        d_usrp->set_rx_gain(opt.rx_gain, opt.rx_channel);
        d_usrp->set_rx_antenna("RX2", opt.rx_channel);
        fprintf(stderr, "radio_uhd: RX ch%d freq=%.3f MHz gain=%.1f dB rate=%.3f Msps\n",
                opt.rx_channel, opt.rx_freq / 1e6, opt.rx_gain, opt.rate / 1e6);
    }

    // Small sleep for LO lock
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // --- Time synchronization (PPS-based, like OAI) ---
    fprintf(stderr, "radio_uhd: waiting for PPS edge...\n");
    uhd::time_spec_t last_pps = d_usrp->get_time_last_pps();
    while (d_usrp->get_time_last_pps() == last_pps)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    d_usrp->set_time_next_pps(uhd::time_spec_t(0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    fprintf(stderr, "radio_uhd: USRP time synced (now=%.3f s)\n",
            d_usrp->get_time_now().get_real_secs());

    if (opt.do_tx) {
        uhd::stream_args_t args("fc32", "sc16");
        args.channels = {(size_t)opt.tx_channel};
        args.args["num_send_frames"] = std::to_string(opt.uhd_num_send_frames);
        if (opt.uhd_send_buff_size > 0)
            args.args["send_buff_size"] = std::to_string(opt.uhd_send_buff_size);
        fprintf(stderr, "radio_uhd: TX stream args: num_send_frames=%d%s\n",
                opt.uhd_num_send_frames,
                opt.uhd_send_buff_size > 0
                    ? (" send_buff_size=" +
                       std::to_string(opt.uhd_send_buff_size)).c_str()
                    : "");
        d_tx = d_usrp->get_tx_stream(args);
    }

    if (opt.do_rx) {
        uhd::stream_args_t args("fc32", "sc16");
        args.channels = {(size_t)opt.rx_channel};
        args.args["num_recv_frames"] = "512";
        d_rx = d_usrp->get_rx_stream(args);
    }

    // Embedded usrpproxy tenant server: shares this same multi_usrp with an
    // external OAI softmodem over a POSIX shm ring. Started after DVB's own
    // streamers are built; it creates its own streamers on the reserved
    // channels when the OAI side issues CMD_INIT. UHD stays inside this
    // plugin (whose binary follows GPLv3 via UHD) — the app never links it.
    if (opt.usrpproxy_enabled) {
        usrpproxy::TenantServer::Config tcfg;
        tcfg.shm_name     = opt.usrpproxy_shm_name;
        tcfg.tx_channels  = opt.usrpproxy_tx_channels;
        tcfg.rx_channels  = opt.usrpproxy_rx_channels;
        tcfg.tx_antenna   = opt.usrpproxy_tx_antenna;
        tcfg.rx_antenna   = opt.usrpproxy_rx_antenna;
        tcfg.tx_core      = opt.usrpproxy_tx_core;
        tcfg.rx_core      = opt.usrpproxy_rx_core;
        tcfg.tx_ring_size = opt.usrpproxy_tx_ring_size;
        tcfg.rx_ring_size = opt.usrpproxy_rx_ring_size;
        fprintf(stderr, "radio_uhd: starting usrpproxy tenant server (shm=%s)\n",
                tcfg.shm_name.c_str());
        d_tenant = std::make_unique<usrpproxy::TenantServer>(d_usrp, tcfg);
        d_tenant->start();
    }

    return true;
}

void uhd_radio_device::start_rx_continuous(double at_time_secs)
{
    uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    cmd.stream_now = false;
    cmd.time_spec  = uhd::time_spec_t(at_time_secs);
    d_rx->issue_stream_cmd(cmd);
}

void uhd_radio_device::stop_rx()
{
    uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    d_rx->issue_stream_cmd(cmd);
}

size_t uhd_radio_device::recv(void* buf, size_t nsamps, radio_rx_meta& md,
                              double timeout)
{
    size_t got = d_rx->recv(buf, nsamps, d_rx_md, timeout);
    switch (d_rx_md.error_code) {
    case uhd::rx_metadata_t::ERROR_CODE_NONE:
        md.error = radio_rx_error::none;    break;
    case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT:
        md.error = radio_rx_error::timeout; break;
    case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW:
        md.error = radio_rx_error::overflow; break;
    default:
        md.error = radio_rx_error::other;
        fprintf(stderr, "radio_uhd: RX error: %s\n",
                d_rx_md.strerror().c_str());
        break;
    }
    return got;
}

size_t uhd_radio_device::send(const void* buf, size_t nsamps,
                              const radio_tx_meta& m, double timeout)
{
    uhd::tx_metadata_t md;
    md.start_of_burst = m.start_of_burst;
    md.end_of_burst   = m.end_of_burst;
    md.has_time_spec  = m.has_time_spec;
    md.time_spec      = uhd::time_spec_t(m.time_spec);
    const void* p = buf ? buf : "";   // valid ptr even for 0-sample EOB
    return d_tx->send(p, nsamps, md, timeout);
}

radio_async_msg uhd_radio_device::poll_tx_async()
{
    radio_async_msg r;
    uhd::async_metadata_t a;
    if (d_tx && d_tx->recv_async_msg(a, 0.0)) {
        r.present = true;
        r.underflow =
            a.event_code == uhd::async_metadata_t::EVENT_CODE_UNDERFLOW ||
            a.event_code ==
                uhd::async_metadata_t::EVENT_CODE_UNDERFLOW_IN_PACKET;
        r.sequence_error =
            a.event_code == uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR ||
            a.event_code ==
                uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST;
        r.time_error =
            a.event_code == uhd::async_metadata_t::EVENT_CODE_TIME_ERROR;
        r.event_code = (unsigned)a.event_code;
        r.time_spec = a.time_spec.get_real_secs();
    }
    return r;
}

} // namespace

extern "C" radio_device* radio_device_create()
{
    return new uhd_radio_device();
}

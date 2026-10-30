/* -*- c++ -*- */
/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compressed-fronthaul radio plugin for dvbs2-uhd-trx (rfnoc-difi FLEX
 * compressed X410 images). SEPARATE from radio_difi (DIFI-compliant) and
 * radio_uhd (stock UHD) — those stay untouched. Like them it is Apache-2.0
 * source, dlopen'd behind radio_device, and links UHD (GPLv3) only in the
 * distributed .so.
 *
 * This driver targets the FLEX1 image (x410_UC_200_flex1), which carries the
 * compressed fronthaul INSIDE standard VITA-49.2 / DIFI data packets in both
 * directions (a candidate DIFI compression-profile framework):
 *   TX (modulation-compressed, Tier-2 generative profile):
 *       ep0 -> difi_deframer0 -> plmod0 -> rrc0 -> DUC -> sb_tx -> radio.
 *       The host wraps packed dvb_plmod frame words in a DIFI data packet; the
 *       deframer strips the 28-byte VRT prologue and swab16's each payload word
 *       (so we big-endian-store each word => plmod receives it native), then
 *       plmod maps indices, inserts pilots, PL-scrambles and applies the
 *       per-frame PAPR backoff.
 *   RX (bit-width compressed):
 *       radio -> sb_rx -> DDC -> iq_compress0 -> difi_rx0 -> ep1.
 *       iq_compress trims sc16->sc8 (MODE 1, Tier-0 native: packed sc8, no
 *       in-band header, width/scale are context) or DPCM (MODE 2, Tier-2:
 *       self-describing 0xC2 udCompHdr + closed-loop residuals), or passes sc16
 *       through (MODE 0, the reset default). The difi framer wraps each
 *       compressed packet in a standard DIFI data packet (28-byte prologue,
 *       PCC advertises the profile). Delivered as raw UDP via the UHD
 *       raw_payload divert; this plugin strips the DIFI prologue and
 *       decompresses back to fc32.  Past the prologue the framer's swab16 makes
 *       each 32-bit compressed word land as plain big-endian bytes.
 * The two switchboards let one TX and one RX chain point at any RF channel;
 * at reset each routes port 0<->0 (= Radio#0 ch0), which is what CAL_LOOPBACK
 * needs, so no config is required for ch0.
 *
 * --usrp args (prefix selects this plugin, see radio_loader):
 *   compfh:addr=<radio mgmt ip>,dest_addr=<host data ip>,dest_port=<udp port>
 *        [,adapter=sfp1][,tx_ant=CAL_LOOPBACK][,rx_ant=CAL_LOOPBACK]
 *        [,tx_radio=R,tx_chan=C][,rx_radio=R,rx_chan=C]
 *        [,rx=plain|trim|dpcm]   RX compression mode (default plain/MODE0)
 *        [,shift=S][,dbits=R]    iq_compress SHIFT / DPCM residual bits
 */

#include "trx/radio_device.h"
#include <atomic>
#include <sstream>
#include "trx/difi_compression.h"
#include <cusat_dsp/difi_packet.h>

#include <uhd/rfnoc_graph.hpp>
#include <uhd/rfnoc/block_id.hpp>
#include <uhd/rfnoc/ddc_block_control.hpp>
#include <uhd/rfnoc/duc_block_control.hpp>
#include <uhd/rfnoc/mb_controller.hpp>
#include <uhd/rfnoc/radio_control.hpp>
#include <uhd/rfnoc/switchboard_block_control.hpp>
#include <uhd/rfnoc/noc_block_base.hpp>
#include <uhd/rfnoc/registry.hpp>
#include <uhd/rfnoc/defaults.hpp>
#include <uhd/stream.hpp>
#include <uhd/types/metadata.hpp>
#include <uhd/types/stream_cmd.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Minimal UHD controller for the iq_compress block (noc 0xC0DEC): poke the
// ctrlport registers (MODE@0, SHIFT@4, CLIP@8 RO, DBITS@12; byte addresses per
// the block's verilog). Registered here so rfnoc_graph::make() attaches it to
// the 0xC0DEC block; the static registrar runs when this .so is dlopen'd.
namespace compfh {
using namespace uhd::rfnoc;
class iqcomp_block_control : public noc_block_base {
public:
    RFNOC_DECLARE_BLOCK(iqcomp_block_control)
    virtual void set_mode(uint32_t mode)   = 0;
    virtual void set_shift(uint32_t shift) = 0;
    virtual void set_dbits(uint32_t dbits) = 0;
    virtual uint32_t get_clip()            = 0;
    virtual void clear_clip() = 0;
};
class iqcomp_block_control_impl : public iqcomp_block_control {
public:
    RFNOC_BLOCK_CONSTRUCTOR(iqcomp_block_control) {}
    void set_mode(uint32_t m)   override { regs().poke32(0,  m); }
    void set_shift(uint32_t s)  override { regs().poke32(4,  s); }
    void set_dbits(uint32_t d)  override { regs().poke32(12, d); }
    uint32_t get_clip()         override { return regs().peek32(8); }
    void clear_clip() override { regs().poke32(8, 0); }
};
UHD_RFNOC_BLOCK_REGISTER_DIRECT(iqcomp_block_control, 0xC0DEC, "IQComp",
                                CLOCK_KEY_GRAPH, "bus_clk")

// The full DIFI block uses its historic word-indexed register map (0..33),
// unlike the byte-addressed IQComp registers. Provision its advertised profile.
class framer_block_control : public noc_block_base {
public:
    RFNOC_DECLARE_BLOCK(framer_block_control)
    virtual void configure(uint16_t pcc, const std::vector<uint32_t>& context, double tick_rate) = 0;
};
class framer_block_control_impl : public framer_block_control {
public:
    RFNOC_BLOCK_CONSTRUCTOR(framer_block_control) {}
    void configure(uint16_t pcc, const std::vector<uint32_t>& context, double tick_rate) override {
        regs().poke32(0, 1); // UTC integer timestamp indicator
        regs().poke32(1, 0x88888888);
        regs().poke32(2, 0x6A621E);
        regs().poke32(3, 0);
        regs().poke32(4, pcc);
        // CHDR timestamps count radio-clock ticks, independently of DDC rate.
        const auto period = static_cast<uint64_t>(std::llround(std::ldexp(1.0,64)/tick_rate));
        regs().poke32(5, static_cast<uint32_t>(period >> 32));
        regs().poke32(6, static_cast<uint32_t>(period));
        for (size_t i=0; i<context.size(); ++i) regs().poke32(7+i,context[i]);
    }
};
UHD_RFNOC_BLOCK_REGISTER_DIRECT(framer_block_control, 0xD1F1, "DIFIFramer",
                                CLOCK_KEY_GRAPH, "bus_clk")

// Controller for the rung-2 RX RRC matched-filter/fractional-resampler block
// (noc 0xB22C2). Byte-addressed ctrlport regs (rrc_rx_regs.vh): RATE_NUM(P)@8,
// RATE_DEN(Q)@12, COEFF_ADDR@20, COEFF_DATA@24 (auto-increments the tap addr).
// The block resamples out/in = P/Q with a coefficient RAM (33 taps x 64
// branches, Q2.14) the DU must load before streaming (the static tap ROM was
// removed in the fractional image), so set_rate + load_coeffs are mandatory.
class rrcrx_block_control : public noc_block_base {
public:
    RFNOC_DECLARE_BLOCK(rrcrx_block_control)
    virtual void set_rate(uint32_t p, uint32_t q)              = 0;
    virtual void load_coeffs(const std::vector<int16_t>& cram) = 0;
    // Rung-3 mode 2 (feedforward energy-search sync): stride = in_sps (FULL
    // symbol, Q8.24), kp = phase-step damping (0 = apply full estimate),
    // ki = rate-integrator shift (stability: 2^-ki * period/stride < 1),
    // estp = log2 estimation period in input samples (>= 11).
    virtual void set_sync_ff(uint32_t stride_q24, uint32_t kp, uint32_t ki,
                             uint32_t estp)                    = 0;
    // Back to rung 2 (mode 0). The sync registers persist across sessions
    // (ctrlport state survives until FPGA reload), so the mode-0 path must
    // clear a mode a previous run may have left armed.
    virtual void set_sync_off()                                = 0;
    // Mode-2 estimator gear (DU coherence gating): live-repoke KP + KI.
    virtual void set_est_gear(uint32_t kp, uint32_t ki)        = 0;
};
class rrcrx_block_control_impl : public rrcrx_block_control {
public:
    RFNOC_BLOCK_CONSTRUCTOR(rrcrx_block_control) {}
    void set_rate(uint32_t p, uint32_t q) override
    {
        regs().poke32(8,  p);   // REG_RRC_RATE_NUM (P)
        regs().poke32(12, q);   // REG_RRC_RATE_DEN (Q, power of two)
    }
    void set_sync_ff(uint32_t stride_q24, uint32_t kp, uint32_t ki,
                     uint32_t estp) override
    {
        regs().poke32(44, stride_q24);  // REG_SYNC_STRIDE (mode 2: full symbol)
        regs().poke32(48, kp);          // REG_SYNC_KP (phase-step damping)
        regs().poke32(52, ki);          // REG_SYNC_KI (rate integrator)
        regs().poke32(64, estp);        // REG_SYNC_ESTP (log2 est period)
        regs().poke32(40, 2);           // REG_SYNC_MODE = 2, armed last
    }
    void set_sync_off() override
    {
        regs().poke32(40, 0);           // REG_SYNC_MODE = 0 (P/Q rung 2)
    }
    void set_est_gear(uint32_t kp, uint32_t ki) override
    {
        regs().poke32(48, kp);          // REG_SYNC_KP
        regs().poke32(52, ki);          // REG_SYNC_KI
    }
    void load_coeffs(const std::vector<int16_t>& cram) override
    {
        regs().poke32(20, 0);   // REG_RRC_COEFF_ADDR = 0 (auto-inc from here)
        for (int16_t v : cram)
            regs().poke32(24, (uint32_t)(uint16_t)v);   // REG_RRC_COEFF_DATA
    }
};
UHD_RFNOC_BLOCK_REGISTER_DIRECT(rrcrx_block_control, 0xB22C2, "RRCRx",
                                CLOCK_KEY_GRAPH, "bus_clk")
} // namespace compfh

namespace {

constexpr size_t DIFI_HDR_BYTES = 28;   // 7 x 32-bit VITA-49.2 / DIFI prologue
constexpr size_t MAX_DGRAM      = 16384;

// Root-raised-cosine impulse response at time t (symbol intervals), rolloff a —
// bit-identical to cusat interpolator.cc / difipack fsplit::rrc_response so the
// RU matched filter matches the TX pulse shape.
double rrc_response(double t, double a)
{
    if (t == 0.0) return 1.0 - a + 4.0 * a / M_PI;
    const double x = 4.0 * a * t;
    if (std::abs(1.0 - x * x) < 1e-9) {
        const double phi = M_PI / (4.0 * a);
        return (a / std::sqrt(2.0)) *
               ((1.0 + 2.0 / M_PI) * std::sin(phi) + (1.0 - 2.0 / M_PI) * std::cos(phi));
    }
    return (std::sin(M_PI * t * (1.0 - a)) + x * std::cos(M_PI * t * (1.0 + a))) /
           (M_PI * t * (1.0 - x * x));
}

// Build the rrc_rx coefficient RAM image (mirror of difipack fsplit
// design_polyphase_rrc + upload_coeffs, the FPGA's golden model). NT=33 fills
// the RAM depth (sub_half=16 => in_sps=4, rrc_delay=4). P polyphase branches,
// fixed stride MAXBR=64, time-reversed taps, Q2.14. Returns NT*MAXBR entries.
std::vector<int16_t> rrcrx_build_coeffs(double in_sps, double rolloff, int P,
                                        int phase_sign)
{
    constexpr int SUB_HALF = 16, NT = 2 * SUB_HALF + 1;   // 33 taps/branch
    constexpr int MAXBR = 64, FRAC = 14;
    std::vector<std::vector<double>> branch(P, std::vector<double>(NT, 0.0));
    for (int b = 0; b < P; b++) {
        // BRANCH PHASE CONVENTION — the two NCOs need OPPOSITE banks:
        //   mode 0 (P/Q accumulate-up): the NCO emits branch b = acc-Q after
        //   consuming input n, where acc = n*P - m*Q = b for the m-th output;
        //   the uniform instant is t_m = m*Q/P = n - b/P, a BACKWARD shift ->
        //   phase_sign = -1, P branches. A +b/P bank staggers the outputs by
        //   up to 2*(P-1)/P samples of cyclic jitter — invisible at integer
        //   ratios (only b=0 fires) but ruinous at any fractional P/Q.
        //   mode 1/2 (Q8.24 countdown): the strobe at sample n with residual
        //   nco = mu targets instant n + mu, a FORWARD shift ->
        //   phase_sign = +1, 64 branches.
        const double frac = phase_sign * (double)b / (double)P;
        double dc = 0.0;
        for (int i = 0; i < NT; i++) {
            const int    d = i - SUB_HALF;                // input tap offset
            const double t = (frac - d) / in_sps;         // symbol-time distance
            branch[b][i] = rrc_response(t, rolloff);
            dc += branch[b][i];
        }
        for (double& v : branch[b]) v /= dc;              // unit DC gain per phase
    }
    std::vector<int16_t> cram((size_t)NT * MAXBR, 0);
    for (int k = 0; k < NT; k++)
        for (int b = 0; b < P; b++) {
            long v = std::lround(branch[b][NT - 1 - k] * (double)(1 << FRAC));
            v = v > 32767 ? 32767 : (v < -32768 ? -32768 : v);
            cram[(size_t)k * MAXBR + b] = (int16_t)v;      // index k*MAXBR + b
        }
    return cram;
}

std::map<std::string, std::string> parse_args(const std::string& s)
{
    std::map<std::string, std::string> kv;
    std::string body = s.rfind("compfh:", 0) == 0 ? s.substr(7) : s;
    size_t pos = 0;
    while (pos < body.size()) {
        size_t comma = body.find(',', pos);
        std::string tok = body.substr(pos, comma == std::string::npos
                                                ? std::string::npos
                                                : comma - pos);
        size_t eq = tok.find('=');
        if (eq != std::string::npos)
            kv[tok.substr(0, eq)] = tok.substr(eq + 1);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return kv;
}

class compfh_radio_device : public radio_device {
public:
  ~compfh_radio_device() override
  {
      if (d_sock >= 0)
          close(d_sock);
      if (d_control >= 0)
          close(d_control);
  }

    bool   setup(const radio_config& cfg) override;
    double get_time_now() override
    {
        return d_tk ? d_tk->get_time_now().get_real_secs() : 0.0;
    }
    void rx_sync_gear(bool locked) override
    {
        if (!d_rrc_ctrl)
            return;
        // Tracking gear: heavier phase-step damping (KP + 3 → D/32 steps at
        // the full estimate rate), KI raised alongside to keep the PI
        // damping margin. The estimator noise floor is the 32-arm grid
        // (stride/32); once locked, small D means the acquisition wrap-trap
        // that forbids weak pulls no longer applies. The estimation PERIOD
        // is deliberately unchanged — rarer estimates just let rate wander
        // accumulate into bigger steps (measured).
        constexpr int GEAR_SHIFT = 3;
        if (locked)
            d_rrc_ctrl->set_est_gear((uint32_t)(d_rrc_kp + GEAR_SHIFT),
                                     (uint32_t)(d_rrc_ki_eff + GEAR_SHIFT));
        else
            d_rrc_ctrl->set_est_gear((uint32_t)d_rrc_kp,
                                     (uint32_t)d_rrc_ki_eff);
    }
    // Compressed packets expand to more samples than their wire size (DPCM ~4x,
    // trim 2x). Bound generously; true N per packet is capped by the DDC spp.
    size_t rx_max_samps() override { return (MAX_DGRAM - DIFI_HDR_BYTES) * 2; }
    size_t tx_max_samps() override { return d_tx_spp > d_tx_hdrw ? (d_tx_spp - d_tx_hdrw) / 2 : 0; }
    void   start_rx_continuous(double at_time_secs) override;
    void   stop_rx() override;
    size_t recv(void* buf, size_t nsamps, radio_rx_meta& md, double timeout) override;
    size_t send(const void* buf, size_t nsamps, const radio_tx_meta& md,
                double timeout) override;
    radio_async_msg poll_tx_async() override;
    radio_live_stats live_stats() const override
    {
        radio_live_stats r;
        r.rx_wire_bytes = d_wire_rx.load();
        r.tx_wire_bytes = d_wire_tx.load();
        r.rx_packets = d_packets.load();
        r.switches = d_switches.load();
        r.clips = d_clips.load();
        const int c = d_live_config.load();
        r.mode = c & 3;
        r.shift = (c >> 4) & 15;
        r.dbits = (c >> 8) & 15;
        return r;
    }
    void poll_control();
    void configure_context();

  private:
    uhd::rfnoc::rfnoc_graph::sptr          d_graph;
    uhd::rfnoc::radio_control::sptr        d_radio;      // RX radio
    uhd::rfnoc::radio_control::sptr        d_radio_tx;   // TX radio
    uhd::rfnoc::mb_controller::timekeeper::sptr d_tk;
    uhd::rx_streamer::sptr                 d_rx;
    uhd::tx_streamer::sptr                 d_tx;
    uhd::rfnoc::block_id_t                 d_framer_id;  // difi framer (RX), 0xD1F1
    uhd::rfnoc::block_id_t                 d_defr_id;    // difi deframer (TX), 0xBD1F2
    uhd::rfnoc::block_id_t                 d_iqcomp_id;

    int d_sock = -1, d_control = -1;
    std::atomic<uint64_t> d_wire_rx{0}, d_wire_tx{0}, d_packets{0}, d_switches{0}, d_clips{0};
    std::atomic<int> d_live_config{0};
    double d_rate = 0, d_freq = 0, d_tick_rate = 0;
    bool     d_issue_on_streamer = false;
    int      d_tx_chan = 0, d_rx_chan = 0;
    int      d_rx_mode = 0;          // 0 passthrough sc16, 1 trim, 2 dpcm
    int      d_rx_shift = 7;         // iq_compress SHIFT (sc16>>SHIFT window)
    int      d_rx_dbits = 6;         // iq_compress DPCM residual bits (mode 2)
    // rrc_rx (0xB22C2) matched-filter fractional resampler: out/in = P/Q. The
    // DDC runs at cfg.rate*Q/P so the host receives cfg.rate after the RU
    // resamples. P=Q=1 (or no block) => not in use.
    uhd::rfnoc::block_id_t d_rrcrx_id;
    int      d_rrc_p = 1, d_rrc_q = 1;
    // Rung-3 mode 2 (rx_symbol_sync): feedforward energy-search sync in the
    // RU; the host receives 1 sample/symbol at cfg.rate == sym_rate.
    bool     d_rrc_ff = false;
    compfh::rrcrx_block_control::sptr d_rrc_ctrl;  // cached for runtime freeze
    double   d_rrc_symrate = 0.0;    // symbol rate (Hz)
    double   d_rrc_insps = 2.0;      // requested RU input sps (DDC out / sym)
    int      d_rrc_kp = 2, d_rrc_ki = 0 /* 0 = auto from estp */, d_rrc_estp = 13;
    int      d_rrc_ki_eff = 18;      // resolved acquisition KI (for gear shift)
    size_t   d_tx_spp  = 0;          // streamer items per packet (sc16)
    size_t   d_tx_hdrw = DIFI_HDR_BYTES / 4;  // DIFI header words per packet
    uint8_t  d_tx_seq  = 0;          // 4-bit DIFI seq for TX headers
    int      d_last_seq = -1;
    std::vector<int16_t> d_tx_scratch;
    uint8_t  d_dgram[MAX_DGRAM];
};

bool compfh_radio_device::setup(const radio_config& cfg)
{
    if (cfg.usrpproxy_enabled) {
        fprintf(stderr, "radio_compfh: usrpproxy not supported\n");
        return false;
    }
    auto kv = parse_args(cfg.usrp_args);
    const std::string addr      = kv.count("addr") ? kv["addr"] : "";
    const std::string dest_addr = kv.count("dest_addr") ? kv["dest_addr"] : "";
    const std::string dest_port = kv.count("dest_port") ? kv["dest_port"] : "7777";
    const std::string adapter   = kv.count("adapter") ? kv["adapter"] : "";
    const std::string tx_ant    = kv.count("tx_ant") ? kv["tx_ant"] : "CAL_LOOPBACK";
    const std::string rx_ant    = kv.count("rx_ant") ? kv["rx_ant"] : "CAL_LOOPBACK";
    d_tx_chan = kv.count("tx_chan") ? std::stoi(kv["tx_chan"]) : 0;
    d_rx_chan = kv.count("rx_chan") ? std::stoi(kv["rx_chan"]) : 0;
    const int tx_blk = kv.count("tx_radio") ? std::stoi(kv["tx_radio"]) : 0;
    const int rx_blk = kv.count("rx_radio") ? std::stoi(kv["rx_radio"]) : 0;
    if (kv.count("rx")) {
        const std::string& r = kv["rx"];
        if (r != "plain" && r != "trim" && r != "dpcm") {
            fprintf(stderr, "radio_compfh: rx must be plain, trim or dpcm\n"); return false;
        }
        d_rx_mode = (r == "trim") ? 1 : (r == "dpcm") ? 2 : 0;
    }
    if (kv.count("shift")) d_rx_shift = std::stoi(kv["shift"]);
    if (kv.count("dbits")) d_rx_dbits = std::stoi(kv["dbits"]);
    if (d_rx_shift < 0 || d_rx_shift > 8 || d_rx_dbits < 1 || d_rx_dbits > 8 ||
        tx_blk < 0 || tx_blk > 1 || rx_blk < 0 || rx_blk > 1 ||
        d_tx_chan < 0 || d_tx_chan > 1 || d_rx_chan < 0 || d_rx_chan > 1) {
        fprintf(stderr, "radio_compfh: invalid shift, residual width or RF channel\n");
        return false;
    }
    if (addr.empty() || (cfg.do_rx && dest_addr.empty())) {
        fprintf(stderr, "radio_compfh: need compfh:addr=<radio ip>,dest_addr="
                "<host data ip>[,dest_port=][,adapter=]\n");
        return false;
    }
    if (cfg.do_tx && cfg.tx_rate <= 0.0) {
        fprintf(stderr, "radio_compfh: needs the app's symbol rate "
                "(--tx-modcomp/--sym-rate sets radio_config.tx_rate)\n");
        return false;
    }

    fprintf(stderr, "radio_compfh: rfnoc graph on addr=%s\n", addr.c_str());
    std::string device_args = "addr=" + addr;
    if (cfg.uhd_send_frame_size > 0) device_args += ",send_frame_size=" + std::to_string(cfg.uhd_send_frame_size);
    if (cfg.uhd_recv_frame_size > 0) device_args += ",recv_frame_size=" + std::to_string(cfg.uhd_recv_frame_size);
    d_graph = uhd::rfnoc::rfnoc_graph::make(device_args);

    // Enumerate blocks: plmod (0xBD5B2), rrc_interp (0xB22C1), iq_compress
    // (0xC0DEC), difi framer (0xD1F1 full / 0xBD1F1 basic) and difi deframer
    // (0xBD1F2) are all generic (identified by NOC ID). FLEX1 has exactly one
    // of each. The framer/deframer are the SEP-facing edges of the two chains.
    std::string plmod, rrc, iqcomp, framer, deframer, rrc_rx;
    for (const auto& id : d_graph->find_blocks("")) {
        const uint32_t noc = d_graph->get_block(id)->get_noc_id();
        fprintf(stderr, "radio_compfh:   block %s (noc_id 0x%X)\n",
                id.to_string().c_str(), noc);
        if (noc == 0xBD5B2) plmod = id.to_string();
        else if (noc == 0xC0DEC) iqcomp = id.to_string();
        else if (noc == 0xB22C1) rrc = id.to_string();      // TX rrc_interp (2x)
        else if (noc == 0xB22C2) rrc_rx = id.to_string();   // RX rrc matched filter+decim
        else if (noc == 0xD1F1) framer = id.to_string();
        else if (noc == 0xBD1F2) deframer = id.to_string();
    }
    // RX functional-split rung 2: the RRC matched-filter/fractional-resampler
    // (0xB22C2, chain ddc -> rrc_rx -> iq_compress -> framer) resamples out/in =
    // P/Q toward symbol-Nyquist, so the DDC must run Q/P x faster for the host
    // to receive cfg.rate. Q is a power of two; the design keeps in_sps=4 (33-tap
    // RAM), so P/Q = out_sps/4 with out_sps = cfg.rate/sym_rate. The coefficient
    // RAM must be loaded before streaming (the static ROM was removed).
    double rrc_in_sps = 0.0, rrc_rolloff = 0.2;
    if (cfg.do_rx && !rrc_rx.empty()) {
        d_rrcrx_id = uhd::rfnoc::block_id_t(rrc_rx);
        // Symbol rate: from tx_rate (set by --tx-modcomp) or an explicit arg.
        const double sym_rate = kv.count("sym_rate") ? std::stod(kv["sym_rate"])
                                                      : cfg.tx_rate;
        rrc_rolloff = kv.count("rolloff") ? std::stod(kv["rolloff"]) : 0.2;
        if (sym_rate <= 0.0) {
            fprintf(stderr, "radio_compfh: rrc_rx needs the symbol rate "
                    "(run with --tx-modcomp, or pass sym_rate= in --usrp args)\n");
            return false;
        }
        d_rrc_symrate = sym_rate;
        if (cfg.rx_symbol_sync) {
            // Rung 3, mode 2: the RU recovers timing and emits 1 sps; the
            // host rate must be the symbol rate. The RU input oversampling
            // (DDC output / sym_rate) is a free choice in [1.2, 7.0] —
            // default 2.0 (61.44 Msps at 30.72 Msym = DDC decim 4).
            if (std::abs(cfg.rate - sym_rate) > 1.0) {
                fprintf(stderr, "radio_compfh: rx_symbol_sync needs the host "
                        "rate == symbol rate (got %.0f vs %.0f)\n",
                        cfg.rate, sym_rate);
                return false;
            }
            d_rrc_ff    = true;
            d_rrc_insps = kv.count("insps") ? std::stod(kv["insps"]) : 2.0;
            d_rrc_kp    = kv.count("kp")    ? std::stoi(kv["kp"])    : 2;
            d_rrc_ki    = kv.count("ki")    ? std::stoi(kv["ki"])    : 0;
            d_rrc_estp  = kv.count("estp")  ? std::stoi(kv["estp"])  : 13;
            if (d_rrc_insps < 1.2 || d_rrc_insps > 7.0 || d_rrc_estp < 11) {
                fprintf(stderr, "radio_compfh: rx_symbol_sync insps must be in "
                        "[1.2, 7.0] and estp >= 11 (got %.3f / %d)\n",
                        d_rrc_insps, d_rrc_estp);
                return false;
            }
            fprintf(stderr, "radio_compfh: rrc_rx mode 2 (feedforward sync) — "
                    "sym_rate %.3f Msym, RU in_sps %.3f, kp %d, estp %d\n",
                    sym_rate / 1e6, d_rrc_insps, d_rrc_kp, d_rrc_estp);
        } else {
            const double out_sps = cfg.rate / sym_rate;   // host samples/symbol
            // Unity resampling uses one branch; avoid an unreduced 16/16 ratio.
            d_rrc_q = kv.count("q") ? std::stoi(kv["q"])
                : (std::abs(out_sps - 4.0) < 1e-6 ? 1 : 16);
            d_rrc_p = kv.count("p") ? std::stoi(kv["p"])
                                    : (int)std::lround(out_sps * d_rrc_q / 4.0);
            if (d_rrc_p < 1) d_rrc_p = 1;
            rrc_in_sps = out_sps * (double)d_rrc_q / (double)d_rrc_p;  // ~4
            fprintf(stderr, "radio_compfh: rrc_rx fractional resample P/Q = %d/%d "
                    "(out_sps %.3f, in_sps %.3f, rolloff %.2f)\n",
                    d_rrc_p, d_rrc_q, out_sps, rrc_in_sps, rrc_rolloff);
        }
    } else if (cfg.do_rx && cfg.rx_symbol_sync) {
        fprintf(stderr, "radio_compfh: rx_symbol_sync requested but the image "
                "has no rrc_rx (0xB22C2) block\n");
        return false;
    }
    if (cfg.do_tx && deframer.empty()) {
        fprintf(stderr, "radio_compfh: no difi_deframer (0xBD1F2) block — not a "
                "DIFI-encapsulated compressed-TX image\n");
        return false;
    }
    if (cfg.do_tx && plmod.empty()) {
        fprintf(stderr, "radio_compfh: no dvb_plmod (0xBD5B2) block — not a "
                "compressed-TX image\n");
        return false;
    }
    if (cfg.do_rx && framer.empty()) {
        fprintf(stderr, "radio_compfh: no difi framer (0xD1F1) block — not a "
                "DIFI-encapsulated compressed-RX image\n");
        return false;
    }
    if (cfg.do_rx && iqcomp.empty()) {
        fprintf(stderr, "radio_compfh: no iq_compress (0xC0DEC) block — not a "
                "compressed-RX image\n");
        return false;
    }
    d_iqcomp_id = uhd::rfnoc::block_id_t(iqcomp);
    if (!deframer.empty()) d_defr_id   = uhd::rfnoc::block_id_t(deframer);
    if (!framer.empty())   d_framer_id = uhd::rfnoc::block_id_t(framer);

    // Resolve the two switchboards from static connections: the one whose
    // output feeds the DDC is sb_rx; the one fed by the DUC is sb_tx. Also
    // learn the port that reaches the chosen radio channel in each.
    auto blkno = [](const std::string& id, const char* name) -> int {
        auto pos = id.find(std::string(name) + "#");
        return pos == std::string::npos
            ? -1 : std::stoi(id.substr(pos + strlen(name) + 1));
    };
    std::string sb_tx, sb_rx;
    int sb_tx_out = 0, sb_rx_in = 0;
    const auto edges = d_graph->enumerate_static_connections();
    for (const auto& e : edges) {
        if (blkno(e.src_blockid, "DUC") >= 0 &&
            blkno(e.dst_blockid, "Switchboard") >= 0)
            sb_tx = e.dst_blockid;                       // DUC -> sb_tx
        if (blkno(e.src_blockid, "Switchboard") >= 0 &&
            blkno(e.dst_blockid, "DDC") >= 0)
            sb_rx = e.src_blockid;                       // sb_rx -> DDC
    }
    for (const auto& e : edges) {
        if (e.src_blockid == sb_tx && blkno(e.dst_blockid, "Radio") == tx_blk &&
            (int)e.dst_port == d_tx_chan)
            sb_tx_out = (int)e.src_port;                 // sb_tx out_N -> radio
        if (e.dst_blockid == sb_rx && blkno(e.src_blockid, "Radio") == rx_blk &&
            (int)e.src_port == d_rx_chan)
            sb_rx_in = (int)e.dst_port;                  // radio -> sb_rx in_N
    }
    fprintf(stderr, "radio_compfh:   TX ep -> %s -> %s -> rrc -> DUC -> "
            "%s[out %d] -> Radio#%d:%d\n", deframer.c_str(), plmod.c_str(),
            sb_tx.c_str(), sb_tx_out, tx_blk, d_tx_chan);
    fprintf(stderr, "radio_compfh:   RX Radio#%d:%d -> %s[in %d] -> DDC -> "
            "%s -> %s -> ep\n", rx_blk, d_rx_chan, sb_rx.c_str(), sb_rx_in,
            iqcomp.c_str(), framer.c_str());

    d_radio = d_graph->get_block<uhd::rfnoc::radio_control>(
        uhd::rfnoc::block_id_t(0, "Radio", rx_blk));
    d_radio_tx = (tx_blk == rx_blk) ? d_radio
        : d_graph->get_block<uhd::rfnoc::radio_control>(
              uhd::rfnoc::block_id_t(0, "Radio", tx_blk));
    const double radio_rate = d_radio->get_rate();
    fprintf(stderr, "radio_compfh: radio rate %.3f Msps\n", radio_rate / 1e6);

    // Point the switchboards at the chosen channel (reset default is 0<->0).
    if (cfg.do_tx && !sb_tx.empty()) {
        auto sb = d_graph->get_block<uhd::rfnoc::switchboard_block_control>(
            uhd::rfnoc::block_id_t(sb_tx));
        sb->connect(0, sb_tx_out);
    }
    if (cfg.do_rx && !sb_rx.empty()) {
        auto sb = d_graph->get_block<uhd::rfnoc::switchboard_block_control>(
            uhd::rfnoc::block_id_t(sb_rx));
        sb->connect(sb_rx_in, 0);
    }

    if (cfg.do_tx) {
        auto duc = d_graph->get_block<uhd::rfnoc::duc_block_control>(
            uhd::rfnoc::block_id_t(0, "DUC", 0));
        d_radio_tx->set_tx_frequency(cfg.tx_freq, d_tx_chan);
        d_radio_tx->set_tx_gain(cfg.tx_gain, d_tx_chan);
        d_radio_tx->set_tx_antenna(tx_ant, d_tx_chan);
        // plmod emits symbol-rate; rrc_interp doubles before the DUC.
        const double duc_in = 2.0 * cfg.tx_rate;
        duc->set_output_rate(radio_rate, 0);
        const double got = duc->set_input_rate(duc_in, 0);
        fprintf(stderr, "radio_compfh: DUC in %.3f Msps (req %.3f) [2x rrc]\n",
                got / 1e6, duc_in / 1e6);
        if (std::abs(got - duc_in) > 1.0) {
            fprintf(stderr, "radio_compfh: DUC cannot make %.3f from %.3f "
                    "(pick sym_rate = %.2fM/2^k)\n", duc_in / 1e6,
                    radio_rate / 1e6, radio_rate / 2e6);
            return false;
        }
        fprintf(stderr, "radio_compfh: TX Radio#%d:%d %.3f MHz gain %.1f ant %s "
                "(modulation-compressed via deframer+plmod+rrc+DUC)\n", tx_blk,
                d_tx_chan, cfg.tx_freq / 1e6, cfg.tx_gain, tx_ant.c_str());
    }
    if (cfg.do_rx) {
        auto ddc = d_graph->get_block<uhd::rfnoc::ddc_block_control>(
            uhd::rfnoc::block_id_t(0, "DDC", 0));
        d_radio->set_rx_frequency(cfg.rx_freq, d_rx_chan);
        d_radio->set_rx_gain(cfg.rx_gain, d_rx_chan);
        d_radio->set_rx_antenna(rx_ant, d_rx_chan);
        ddc->set_input_rate(radio_rate, 0);
        // With rrc_rx in the chain the DDC runs faster than the host rate:
        // mode 0 by Q/P (the RU resamples P/Q down to cfg.rate), mode 2 by
        // in_sps (the RU emits 1 sample/symbol at cfg.rate == sym_rate).
        const bool have_rrcrx = !d_rrcrx_id.to_string().empty() && d_rrc_p >= 1;
        const double ddc_req = !have_rrcrx ? cfg.rate
            : (d_rrc_ff ? d_rrc_insps * d_rrc_symrate
                        : cfg.rate * (double)d_rrc_q / (double)d_rrc_p);
        const double got = ddc->set_output_rate(ddc_req, 0);
        if (have_rrcrx && d_rrc_ff)
            fprintf(stderr, "radio_compfh: DDC out %.3f Msps (req %.3f) -> rrc_rx "
                    "mode 2 -> host %.3f Msym at 1 sps [rung-3 timing split]\n",
                    got / 1e6, ddc_req / 1e6, d_rrc_symrate / 1e6);
        else if (have_rrcrx)
            fprintf(stderr, "radio_compfh: DDC out %.3f Msps (req %.3f) -> rrc_rx "
                    "*%d/%d -> host %.3f Msps [rung-2 matched-filter split]\n",
                    got / 1e6, ddc_req / 1e6, d_rrc_p, d_rrc_q,
                    got * d_rrc_p / d_rrc_q / 1e6);
        else
            fprintf(stderr, "radio_compfh: DDC out %.3f Msps (req %.3f)\n",
                    got / 1e6, cfg.rate / 1e6);
        // Provision the RU resampler (coefficient RAM upload is mandatory —
        // it powers up with empty taps). in_sps from the negotiated DDC rate.
        if (have_rrcrx && d_rrc_ff) {
            // Mode 2: the NCO dead-reckons at STRIDE between estimates, so the
            // stride must match the ACTUAL DDC rate — a requested-vs-coerced
            // mismatch would read as a constant clock offset (the estimator's
            // aliasing bound is only ~stride/(2*period) ~ 100 ppm).
            const double insps = got / d_rrc_symrate;
            if (insps < 1.2 || insps > 7.0) {
                fprintf(stderr, "radio_compfh: coerced DDC rate gives in_sps "
                        "%.4f outside rrc_rx mode-2 [1.2, 7.0]\n", insps);
                return false;
            }
            const uint32_t stride = (uint32_t)std::llround(insps * 16777216.0);
            // Rate-integrator stability: 2^-ki * (period/stride) < 1. Auto
            // (ki=0): one power of two above the bound.
            // Auto KI: the designer's stability rule (+1) plus 2*KP of PI
            // damping margin — a hotter integrator pumps up on correlated
            // acquisition-transient errors and rails (measured OTA: the
            // rule-minimum KI never locks at kp>=2).
            int ki = d_rrc_ki > 0 ? d_rrc_ki
                   : (int)std::ceil(std::log2((double)(1u << d_rrc_estp) / insps))
                     + 1 + 2 * d_rrc_kp;
            d_rrc_ki_eff = ki;
            auto rrc = d_graph->get_block<compfh::rrcrx_block_control>(d_rrcrx_id);
            const auto cram = rrcrx_build_coeffs(insps, rrc_rolloff, 64, +1);
            rrc->load_coeffs(cram);
            rrc->set_sync_ff(stride, (uint32_t)d_rrc_kp, (uint32_t)ki,
                             (uint32_t)d_rrc_estp);
            d_rrc_ctrl = rrc;  // runtime freeze hook (rx_sync_freeze)
            fprintf(stderr, "radio_compfh: rrc_rx mode 2 provisioned (in_sps "
                    "%.4f, stride 0x%08x, kp %d, ki %d, estp %d, %zu coeff "
                    "words +b/64)\n", insps, stride, d_rrc_kp, ki, d_rrc_estp,
                    cram.size());
        } else if (have_rrcrx) {
            auto rrc = d_graph->get_block<compfh::rrcrx_block_control>(d_rrcrx_id);
            rrc->set_sync_off();  // clear a mode left armed by a previous run
            rrc->set_rate((uint32_t)d_rrc_p, (uint32_t)d_rrc_q);
            const auto cram = rrcrx_build_coeffs(rrc_in_sps, rrc_rolloff,
                                                 d_rrc_p, -1);
            rrc->load_coeffs(cram);
            fprintf(stderr, "radio_compfh: rrc_rx provisioned (P=%d Q=%d, %zu "
                    "coeff words uploaded)\n", d_rrc_p, d_rrc_q, cram.size());
        }
        // Configure iq_compress: MODE 0 is the reset default (passthrough, no
        // poke needed); MODE 1 trim / 2 dpcm need MODE + SHIFT (+ DBITS) set
        // over ctrlport. Trim is Tier-0 native (no in-band header), so the host
        // must know SHIFT out of band — we set it here and use it in recv().
        { // Reset MODE on every run; compression registers survive sessions.
            auto iqc = d_graph->get_block<compfh::iqcomp_block_control>(d_iqcomp_id);
            iqc->set_shift((uint32_t)d_rx_shift);
            if (d_rx_mode == 2) iqc->set_dbits((uint32_t)d_rx_dbits);
            iqc->set_mode((uint32_t)d_rx_mode);
            iqc->clear_clip();
            fprintf(stderr, "radio_compfh: iq_compress MODE %d SHIFT %d%s\n",
                    d_rx_mode, d_rx_shift,
                    d_rx_mode == 2 ? (" DBITS " + std::to_string(d_rx_dbits)).c_str()
                                   : "");
        }
        d_rate = cfg.rate;
        d_freq = cfg.rx_freq;
        d_tick_rate = radio_rate;
        configure_context();
        d_live_config.store(d_rx_mode | (d_rx_shift << 4) | (d_rx_dbits << 8));
        if (cfg.demo_control_enabled) {
            d_control = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
            sockaddr_in control{};
            control.sin_family = AF_INET;
            control.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            control.sin_port = htons(5558);
            if (d_control < 0 || bind(d_control, (sockaddr *)&control, sizeof(control)) < 0)
                throw std::runtime_error("cannot bind loopback compression control port 5558");
        }
        fprintf(stderr, "radio_compfh: RX Radio#%d:%d %.3f MHz gain %.1f ant %s "
                "-> iq_compress(MODE %d) -> difi framer -> UDP %s:%s\n", rx_blk,
                d_rx_chan, cfg.rx_freq / 1e6, cfg.rx_gain, rx_ant.c_str(),
                d_rx_mode, dest_addr.c_str(), dest_port.c_str());

        d_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (d_sock < 0) { perror("radio_compfh: socket"); return false; }
        int rcvbuf = 64 << 20;
        setsockopt(d_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port   = htons((uint16_t)std::stoi(dest_port));
        if (inet_pton(AF_INET, dest_addr.c_str(), &sa.sin_addr) != 1 ||
            bind(d_sock, (sockaddr*)&sa, sizeof(sa)) < 0) {
            perror("radio_compfh: bind (is dest_addr this host's data IP?)");
            return false;
        }
        // raw_payload divert: the difi framer's DIFI packets fire straight to
        // dest_addr:dest_port. Attach the RX streamer at the framer output (the
        // SEP-facing edge of the compressed-RX chain).
        uhd::stream_args_t rargs("sc16", "sc16");
        rargs.args["dest_addr"] = dest_addr;
        rargs.args["dest_port"] = dest_port;
        if (!adapter.empty()) rargs.args["adapter"] = adapter;
        rargs.args["stream_mode"] =
            kv.count("stream_mode") ? kv["stream_mode"] : "raw_payload";
        if (kv.count("dest_mac_addr")) rargs.args["dest_mac_addr"] = kv["dest_mac_addr"];
        d_rx = d_graph->create_rx_streamer(1, rargs);
        d_graph->connect(d_framer_id, 0, d_rx, 0);
        // Stream actions must pass through the DDC and switchboard controllers.
        d_issue_on_streamer = !kv.count("issue") || kv["issue"] != "radio";
    }

    if (cfg.do_tx) {
        // Enter the TX chain at the DIFI deframer. send() accounts for UHD's
        // sc16 packing so the deframer's swab16 recovers native frame words.
        uhd::stream_args_t targs("sc16", "sc16");
        targs.args["num_send_frames"] = std::to_string(cfg.uhd_num_send_frames);
        if (cfg.uhd_send_buff_size > 0)
            targs.args["send_buff_size"] = std::to_string(cfg.uhd_send_buff_size);
        d_tx = d_graph->create_tx_streamer(1, targs);
        d_graph->connect(d_tx, 0, d_defr_id, 0);
    }

    // Static FPGA wiring still needs logical RFNoC edges for stream actions
    // and property propagation. Connect the selected switchboard ports only.
    std::set<std::string> blocks;
    for (const auto& id : d_graph->find_blocks("")) blocks.insert(id.to_string());
    for (const auto& e : edges) {
        if (!blocks.count(e.src_blockid) || !blocks.count(e.dst_blockid)) continue;
        const int src_radio = blkno(e.src_blockid,"Radio");
        const int dst_radio = blkno(e.dst_blockid,"Radio");
        if (src_radio >= 0 && (!cfg.do_rx || src_radio != rx_blk || (int)e.src_port != d_rx_chan)) continue;
        if (dst_radio >= 0 && (!cfg.do_tx || dst_radio != tx_blk || (int)e.dst_port != d_tx_chan)) continue;
        d_graph->connect(uhd::rfnoc::block_id_t(e.src_blockid),e.src_port,
                         uhd::rfnoc::block_id_t(e.dst_blockid),e.dst_port);
    }
    d_graph->commit();
    // Keep RX packet boundaries aligned with the DDC decimation.
    if (cfg.do_rx) d_radio->set_property<int>("spp",256,d_rx_chan);

    if (cfg.do_tx) {
        d_tx_spp = d_tx->get_max_num_samps();
        if (d_tx_spp <= d_tx_hdrw + 1) {
            fprintf(stderr, "radio_compfh: TX packet capacity too small\n"); return false;
        }
        // The deframer strips d_tx_hdrw header words; keep the post-strip
        // payload even (stock DUC pads odd packets otherwise). plmod repacks
        // our words into its own SPP_OUT symbol packets, so this only affects
        // our input packetization.
        if ((d_tx_spp - d_tx_hdrw) & 1) d_tx_spp -= 1;
        d_tx_scratch.resize(d_tx_spp * 2);
        fprintf(stderr, "radio_compfh: TX %zu items/packet (%zu payload words + "
                "%zu DIFI header words)\n", d_tx_spp, d_tx_spp - d_tx_hdrw,
                d_tx_hdrw);
    }
    d_tk = d_graph->get_mb_controller(0)->get_timekeeper(0);
    d_tk->set_time_now(uhd::time_spec_t(0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    fprintf(stderr, "radio_compfh: timekeeper zeroed (now=%.3f s)\n", get_time_now());
    return true;
}

void compfh_radio_device::configure_context()
{
    uint8_t b[cusat_dsp::difi::CONTEXT_PACKET_BYTES];
    cusat_dsp::difi::pack_context(b, 0x88888888, 0, 0, 0, d_rate, d_freq, d_rate);
    if (d_rx_mode != 0) {
        cusat_dsp::difi::put_u32(b + 25 * 4, (1u << 31) | (1u << 29) | (7u << 6) | 7u);
        cusat_dsp::difi::put_u32(b + 18 * 4,
                                 (uint32_t)(uint16_t)std::lround(6.020599913 * d_rx_shift * 128));
    }
    std::vector<uint32_t> words;
    for (size_t i = 0; i < 27; ++i)
        words.push_back(cusat_dsp::difi::get_u32(b + i * 4));
    d_graph->get_block<compfh::framer_block_control>(d_framer_id)
        ->configure(d_rx_mode == 2 ? difipack::compression::PCC_DPCM : 0, words, d_tick_rate);
}

void compfh_radio_device::poll_control()
{
    if (d_control < 0)
        return;
    if ((d_packets.load(std::memory_order_relaxed) & 0xffff) == 0) {
        try {
            d_clips.store(
                d_graph->get_block<compfh::iqcomp_block_control>(d_iqcomp_id)->get_clip());
        } catch (const std::exception &) {
        } // Do not stop sample delivery for a diagnostic read.
    }
    char b[128];
    sockaddr_in peer{};
    socklen_t plen = sizeof(peer);
    const auto len = recvfrom(d_control, b, sizeof(b) - 1, MSG_DONTWAIT, (sockaddr *)&peer, &plen);
    if (len <= 0)
        return;
    b[len] = 0;
    std::istringstream in(b);
    std::string op, extra;
    int mode = -1, shift = -1, bits = -1;
    in >> op >> mode >> shift >> bits;
    std::string response = "{\"ok\":false,\"error\":\"invalid compression request\"}";
    if (in && !(in >> extra) && op == "compression" && mode >= 0 && mode <= 2 && shift >= 0 &&
        shift <= 8 && (bits == 2 || bits == 4 || bits == 6 || bits == 8)) {
        // Only RX owns the decoder and these register writes. Halt RF sample
        // production and drain the old format before changing SHIFT/DBITS:
        // those RTL registers are not latched for the whole payload.
        const int old_mode = d_rx_mode, old_shift = d_rx_shift, old_bits = d_rx_dbits;
        try {
            stop_rx();
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            bool quiet = false;
            while (std::chrono::steady_clock::now() < limit) {
                pollfd p{d_sock, POLLIN, 0};
                if (poll(&p, 1, 50) <= 0) {
                    quiet = true;
                    break;
                }
                ssize_t drained;
                while ((drained = ::recv(d_sock, d_dgram, sizeof(d_dgram), MSG_DONTWAIT)) > 0) {
                    d_wire_rx.fetch_add(drained, std::memory_order_relaxed);
                    d_packets.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!quiet)
                throw std::runtime_error("RX did not drain; compression unchanged");
            auto iqc = d_graph->get_block<compfh::iqcomp_block_control>(d_iqcomp_id);
            iqc->set_shift(shift);
            iqc->set_dbits(bits);
            iqc->set_mode(mode);
            iqc->clear_clip();
            d_clips.store(0);
            d_rx_mode = mode;
            d_rx_shift = shift;
            d_rx_dbits = bits;
            configure_context();
            d_last_seq = -1;
            d_live_config.store(mode | (shift << 4) | (bits << 8));
            start_rx_continuous(get_time_now() + 0.05);
            d_switches.fetch_add(1);
            response = "{\"ok\":true}";
        } catch (const std::exception &e) {
            fprintf(stderr, "radio_compfh: live switch failed: %s\n", e.what());
            // Restore the decoder and every format register together. Leave
            // RX stopped if rollback fails instead of decoding an unknown format.
            try {
                auto iqc = d_graph->get_block<compfh::iqcomp_block_control>(d_iqcomp_id);
                iqc->set_shift(old_shift);
                iqc->set_dbits(old_bits);
                iqc->set_mode(old_mode);
                d_rx_mode = old_mode;
                d_rx_shift = old_shift;
                d_rx_dbits = old_bits;
                configure_context();
                d_last_seq = -1;
                d_live_config.store(old_mode | (old_shift << 4) | (old_bits << 8));
                start_rx_continuous(get_time_now() + 0.05);
            } catch (const std::exception &rollback_error) {
                fprintf(stderr, "radio_compfh: rollback failed, RX left stopped: %s\n",
                        rollback_error.what());
                try {
                    stop_rx();
                } catch (...) {
                }
            }
            response = "{\"ok\":false,\"error\":\"radio switch failed; inspect modem log\"}";
        }
    }
    sendto(d_control, response.data(), response.size(), 0, (sockaddr *)&peer, plen);
}

void compfh_radio_device::start_rx_continuous(double at_time_secs)
{
    uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    cmd.stream_now = false;
    cmd.time_spec  = uhd::time_spec_t(at_time_secs);
    if (d_issue_on_streamer) d_rx->issue_stream_cmd(cmd);
    else                     d_radio->issue_stream_cmd(cmd, d_rx_chan);
}

void compfh_radio_device::stop_rx()
{
    uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    if (d_issue_on_streamer) d_rx->issue_stream_cmd(cmd);
    else                     d_radio->issue_stream_cmd(cmd, d_rx_chan);
}

size_t compfh_radio_device::recv(void* buf, size_t nsamps, radio_rx_meta& md,
                                 double timeout)
{
    poll_control();
    md.error = radio_rx_error::none;
    pollfd pfd{d_sock, POLLIN, 0};
    int pr = poll(&pfd, 1, (int)(timeout * 1000));
    if (pr <= 0) { md.error = radio_rx_error::timeout; return 0; }
    ssize_t len = ::recv(d_sock, d_dgram, sizeof(d_dgram), 0);
    if (len > 0) {
        d_wire_rx.fetch_add(len, std::memory_order_relaxed);
        d_packets.fetch_add(1, std::memory_order_relaxed);
    }
    if (len <= (ssize_t)DIFI_HDR_BYTES) { md.error = radio_rx_error::other; return 0; }

    // The difi framer wraps every compressed packet in a standard DIFI data
    // packet and interleaves periodic Standard Flow Signal Context packets on
    // the same flow — keep signal-data packets only (type nibble 0x1).
    if ((d_dgram[0] >> 4) != 0x1) return 0;
    // 4-bit per-stream sequence: a gap means the NIC/socket dropped datagrams.
    const int seq = d_dgram[1] & 0x0F;
    if (d_last_seq >= 0 && seq != ((d_last_seq + 1) & 0x0F))
        md.error = radio_rx_error::overflow;
    d_last_seq = seq;

    // Past the 28-byte prologue the payload is a plain big-endian byte stream:
    // the framer swab16's each 32-bit word and the raw_payload wire serializes
    // that, so each original iq_compress word W lands as bytes [W3,W2,W1,W0].
    const size_t declared = ((size_t)d_dgram[2] << 8 | d_dgram[3]) * 4;
    if (declared != (size_t)len || !(d_dgram[0] & 8)) {
        md.error = radio_rx_error::other; return 0;
    }
    try {
        return difipack::compression::decode_fc32(d_dgram + DIFI_HDR_BYTES,
            (size_t)len - DIFI_HDR_BYTES, reinterpret_cast<std::complex<float>*>(buf),
            nsamps, d_rx_mode, d_rx_shift);
    } catch (const std::invalid_argument&) {
        md.error = radio_rx_error::other; return 0;
    }
}

size_t compfh_radio_device::send(const void* buf, size_t nsamps,
                                 const radio_tx_meta& m, double timeout)
{
    // Modulation-compressed TX (Tier-2 generative profile): the app hands
    // packed dvb_plmod frame words (2 uint32 per complexf ring slot). Wrap them
    // in a DIFI data packet [28B prologue][big-endian words] and forward through
    // the difi_deframer — it strips our header and swab16's each payload word,
    // so big-endian-storing each word makes plmod receive it native (calibrated
    // against the deframer's validated sc16 byte order). The words are opaque to
    // the deframer (treated as samples); plmod parses prologue+PLHEADER+bits and
    // generates the symbols, emitting its own SPP_OUT packets downstream.
    const size_t HDRW = d_tx_hdrw;                  // 7 DIFI header words
    const size_t payload_cap = d_tx_spp - HDRW;     // plmod words per packet
    const auto* w = reinterpret_cast<const uint32_t*>(buf);
    const size_t total = nsamps ? nsamps * 2 : 1;   // words (>=1 for bare EOB)
    size_t done = 0;
    while (done < total) {
        const size_t n = std::min(payload_cap, total - done);
        uint8_t* h = reinterpret_cast<uint8_t*>(d_tx_scratch.data());
        memset(h, 0, DIFI_HDR_BYTES);
        h[0] = 0x18;                                 // DIFI data pkt (type nibble 1)
        h[1] = 0x60 | (d_tx_seq++ & 0x0F);           // TSI/TSF + 4-bit seq
        h[9] = 0x6A; h[10] = 0x62; h[11] = 0x1E;
        h[14] = 0xDC; h[15] = 0x81; // candidate TX modulation profile
        const uint16_t words = (uint16_t)(n + HDRW);
        h[2] = (words >> 8) & 0xFF; h[3] = words & 0xFF;
        h[4] = 0x88; h[5] = 0x88; h[6] = 0x88; h[7] = 0x88;   // stream id
        uint8_t* pb = reinterpret_cast<uint8_t*>(d_tx_scratch.data() + HDRW * 2);
        for (size_t i = 0; i < n; i++) {
            const auto sample = difipack::compression::uhd_input_word(
                (buf && nsamps) ? w[done + i] : 0u);
            // UHD sc16 -> item32 packs (I << 16) | Q; this is not a
            // byte-transparent streamer. Account for it before deframing.
            memcpy(pb + 4*i, &sample, sizeof(sample));
        }
        uhd::tx_metadata_t md;
        md.start_of_burst = m.start_of_burst && done == 0;
        md.end_of_burst   = m.end_of_burst && (done + n == total);
        md.has_time_spec  = false;                   // untimed stream
        const size_t sent = d_tx->send(d_tx_scratch.data(), n + HDRW, md, timeout);
        if (sent < n + HDRW) break;                  // timeout/backpressure
        d_wire_tx.fetch_add((n + HDRW) * 4, std::memory_order_relaxed);
        done += n;
    }
    return nsamps ? done / 2 : 0;                     // complexf ring slots consumed
}

radio_async_msg compfh_radio_device::poll_tx_async()
{
    radio_async_msg r;
    uhd::async_metadata_t a;
    if (d_tx && d_tx->recv_async_msg(a, 0.0)) {
        r.present    = true;
        r.underflow  = (a.event_code == uhd::async_metadata_t::EVENT_CODE_UNDERFLOW);
        r.event_code = (unsigned)a.event_code;
        r.time_spec  = a.time_spec.get_real_secs();
    }
    return r;
}

} // namespace

extern "C" radio_device* radio_device_create()
{
    return new compfh_radio_device();
}

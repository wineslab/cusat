/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * DIFI radio plugin for dvbs2-uhd-trx (rfnoc-difi X410 images).
 *
 * This source file is Apache-2.0. It links UHD (GPLv3) for the control
 * plane, so a distributed libradio_difi.so binary is governed by GPLv3 as a
 * combined work; like radio_uhd it is dlopen'd behind radio_device so UHD
 * never links into the Apache-2.0 application.
 *
 * Data plane (DIFI/IEEE-ISTO 4900 v1.1, big-endian on the wire):
 *   RX — the FPGA difi framer prepends a 28-byte DIFI signal-data header
 *        (real timestamps, DIFI OUI/class codes) and interleaves periodic
 *        Standard Flow Signal Context packets; UHD 4.8 remote streaming
 *        (raw_payload divert) fires it all as raw UDP at dest_addr:dest_port.
 *        This plugin binds the socket, keeps data packets (type nibble 0x1,
 *        skipping context), strips the header, converts BE sc16 -> fc32.
 *   TX — with tx=difi, frames [28B DIFI header][BE sc16] one DIFI packet per
 *        CHDR packet into the difi_deframer chain (the FPGA strips the
 *        header and byte-swaps to CHDR order before the DUC/radio). Plain
 *        CHDR TX (fc32 via DUC) remains the default.
 * Control plane: UHD RFNoC graph (radio tune/gain, DDC/DUC rates, stream
 * commands — issued on the radio block; framer/deframer instances resolved
 * per channel from the image's static connections).
 *
 * --usrp args (prefix selects this plugin, see radio_loader):
 *   difi:addr=<radio mgmt ip>,dest_addr=<host data ip>,dest_port=<udp port>
 *        [,adapter=sfp1][,tx_ant=TX/RX0][,rx_ant=RX1][,iq=iq|qi]
 *        [,tx=difi[,tx_radio=R,tx_chan=C]]  TX through the deframer chain
 *        [,rx=plain[,rx_radio=R,rx_chan=C]] RX a stock chain via divert (raw
 *                     LE sc16, no DIFI header) — closed-loop debug partner
 *        [,rx_radio=R,rx_chan=C]            pick the DIFI framer chain
 */

#include "trx/radio_device.h"

#include <uhd/rfnoc_graph.hpp>
#include <uhd/rfnoc/block_id.hpp>
#include <uhd/rfnoc/ddc_block_control.hpp>
#include <uhd/rfnoc/duc_block_control.hpp>
#include <uhd/rfnoc/mb_controller.hpp>
#include <uhd/rfnoc/radio_control.hpp>
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
#include <utility>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t DIFI_HDR_BYTES = 28;   // 7 x 32-bit words (difi_basic)
constexpr size_t MAX_DGRAM      = 16384;

/* "difi:k=v,k=v" -> map (the "difi:" prefix is optional here). */
std::map<std::string, std::string> parse_args(const std::string& s)
{
    std::map<std::string, std::string> kv;
    std::string body = s.rfind("difi:", 0) == 0 ? s.substr(5) : s;
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

class difi_radio_device : public radio_device {
public:
    ~difi_radio_device() override
    {
        if (d_sock >= 0) close(d_sock);
    }

    bool   setup(const radio_config& cfg) override;
    double get_time_now() override
    {
        return d_tk ? d_tk->get_time_now().get_real_secs() : 0.0;
    }
    size_t rx_max_samps() override { return (MAX_DGRAM - DIFI_HDR_BYTES) / 4; }
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
    uhd::rfnoc::rfnoc_graph::sptr          d_graph;
    uhd::rfnoc::radio_control::sptr        d_radio;     // RX-side radio
    uhd::rfnoc::radio_control::sptr        d_radio_tx;  // TX-side radio
    uhd::rfnoc::ddc_block_control::sptr    d_ddc;
    uhd::rfnoc::block_id_t                 d_difi_id;   // framer (RX), 0xBD1F1
    uhd::rfnoc::block_id_t                 d_defr_id;   // deframer (TX), 0xBD1F2
    uhd::rfnoc::mb_controller::timekeeper::sptr d_tk;
    uhd::rx_streamer::sptr                 d_rx;   // remote (raw_payload)
    uhd::tx_streamer::sptr                 d_tx;   // CHDR TX (fc32 or sc16)

    int      d_sock = -1;
    bool     d_iq_swapped = false;   // iq=qi -> parse pairs as (Q,I); default iq
    bool     d_issue_on_streamer = false;
    int      d_tx_chan = 0, d_rx_chan = 0;
    size_t   d_rx_hdr = DIFI_HDR_BYTES; // 0 when rx=plain (no DIFI framer in chain)
    bool     d_tx_difi = false;      // tx=difi -> DIFI-in-CHDR via the deframer
    bool     d_tx_sym = false;       // tx=sym  -> symbol-rate DIFI via deframer->rrc_interp
    bool     d_tx_modcomp = false;   // tx=modcomp -> packed plmod frame words via deframer->plmod->rrc
    bool     d_tx_raw = false;       // send() payload is packed uint32 words, not fc32 samples
    bool     d_tx_manual = false;    // manual sc16 framing (tx=difi/sym/sc16/modcomp)
    size_t   d_tx_hdrw = 0;          // header words prepended per packet
    size_t   d_tx_spp = 0;           // streamer items per packet (sc16)
    uint8_t  d_tx_seq = 0;           // 4-bit DIFI seq for TX headers
    std::vector<int16_t> d_tx_scratch;
    int      d_last_seq = -1;
    uint8_t  d_dgram[MAX_DGRAM];
};

bool difi_radio_device::setup(const radio_config& cfg)
{
    if (cfg.usrpproxy_enabled) {
        fprintf(stderr, "radio_difi: usrpproxy (OAI IQ passthrough) is not "
                "supported in DIFI mode\n");
        return false;
    }

    auto kv = parse_args(cfg.usrp_args);
    const std::string addr      = kv.count("addr") ? kv["addr"] : "";
    const std::string dest_addr = kv.count("dest_addr") ? kv["dest_addr"] : "";
    const std::string dest_port = kv.count("dest_port") ? kv["dest_port"] : "7777";
    const std::string adapter   = kv.count("adapter") ? kv["adapter"] : "";
    const std::string tx_ant    = kv.count("tx_ant") ? kv["tx_ant"] : "TX/RX0";
    const std::string rx_ant    = kv.count("rx_ant") ? kv["rx_ant"] : "RX1";
    d_iq_swapped = kv.count("iq") && kv["iq"] == "qi";
    if (addr.empty() || (cfg.do_rx && dest_addr.empty())) {
        fprintf(stderr, "radio_difi: need difi:addr=<radio ip>,dest_addr="
                "<host data ip>[,dest_port=][,adapter=]\n");
        return false;
    }

    fprintf(stderr, "radio_difi: rfnoc graph on addr=%s (control plane)\n",
            addr.c_str());
    d_graph = uhd::rfnoc::rfnoc_graph::make("addr=" + addr);

    // Enumerate blocks; the DIFI block has no registered controller so it
    // shows up under a generic ID — identify by NOC ID (0xBD1F1 basic /
    // 0xD1F1 full). It is a bump-in-the-wire: nothing to configure (the
    // basic variant never acks ctrlport writes, so do NOT poke it).
    std::set<std::string> framer_ids, deframer_ids, rrc_ids, plmod_ids;
    for (const auto& id : d_graph->find_blocks("")) {
        auto blk = d_graph->get_block(id);
        const uint32_t noc_id = blk->get_noc_id();
        fprintf(stderr, "radio_difi:   block %s (noc_id 0x%X)\n",
                id.to_string().c_str(), noc_id);
        if (noc_id == 0xBD1F1 || noc_id == 0xD1F1)
            framer_ids.insert(id.to_string());
        else if (noc_id == 0xBD1F2)
            deframer_ids.insert(id.to_string());
        else if (noc_id == 0xB22C1)   // rrc_interp (2x FPGA pulse shaping)
            rrc_ids.insert(id.to_string());
        else if (noc_id == 0xBD5B2)   // dvb_plmod (modulation-compressed TX)
            plmod_ids.insert(id.to_string());
    }
    // Resolve which framer/deframer serves which (radio_block, channel) from
    // the image's static connections: DDC#R:c -> framer (RX), and
    // deframer -> DUC#R:c (TX). Works for the single-instance images and the
    // ALLDIFI core (one of each per channel) alike.
    auto blkno = [](const std::string& id, const char* name) -> int {
        auto pos = id.find(std::string(name) + "#");
        return pos == std::string::npos
            ? -1 : std::stoi(id.substr(pos + strlen(name) + 1));
    };
    std::map<std::pair<int,int>, std::string> framer_at, deframer_at, sym_at,
        modcomp_at;
    std::map<std::string, std::string> rrc_src;        // rrc id -> feeding deframer
    std::map<std::string, std::string> plmod_src;      // plmod id -> feeding deframer
    std::map<std::string, std::string> rrc_from_plmod; // rrc id -> deframer (via plmod)
    const auto edges = d_graph->enumerate_static_connections();
    for (const auto& e : edges) {
        const int src_ddc = blkno(e.src_blockid, "DDC");
        const int dst_duc = blkno(e.dst_blockid, "DUC");
        if (src_ddc >= 0 && framer_ids.count(e.dst_blockid))
            framer_at[{src_ddc, (int)e.src_port}] = e.dst_blockid;
        if (dst_duc >= 0 && deframer_ids.count(e.src_blockid))
            deframer_at[{dst_duc, (int)e.dst_port}] = e.src_blockid;
        if (rrc_ids.count(e.dst_blockid) && deframer_ids.count(e.src_blockid))
            rrc_src[e.dst_blockid] = e.src_blockid;      // deframer -> rrc (sym chain)
        if (plmod_ids.count(e.dst_blockid) && deframer_ids.count(e.src_blockid))
            plmod_src[e.dst_blockid] = e.src_blockid;    // deframer -> plmod (modcomp)
    }
    // plmod -> rrc: trace the rrc back through the plmod to its feeding deframer.
    for (const auto& e : edges)
        if (rrc_ids.count(e.dst_blockid) && plmod_ids.count(e.src_blockid) &&
            plmod_src.count(e.src_blockid))
            rrc_from_plmod[e.dst_blockid] = plmod_src[e.src_blockid];
    // Second pass: rrc -> DUC#R:c closes the symbol-TX chain (deframer ->
    // rrc_interp -> DUC) and the modcomp chain (deframer -> plmod ->
    // rrc_interp -> DUC), each keyed like deframer_at (on the deframer id).
    for (const auto& e : edges) {
        const int dst_duc = blkno(e.dst_blockid, "DUC");
        if (dst_duc < 0 || !rrc_ids.count(e.src_blockid)) continue;
        if (rrc_from_plmod.count(e.src_blockid))
            modcomp_at[{dst_duc, (int)e.dst_port}] = rrc_from_plmod[e.src_blockid];
        else if (rrc_src.count(e.src_blockid))
            sym_at[{dst_duc, (int)e.dst_port}] = rrc_src[e.src_blockid];
    }
    for (const auto& [rc, id] : modcomp_at)
        fprintf(stderr, "radio_difi:   modcomp  %s -> plmod -> rrc -> Radio#%d:%d\n",
                id.c_str(), rc.first, rc.second);
    for (const auto& [rc, id] : sym_at)
        fprintf(stderr, "radio_difi:   sym-TX   %s -> rrc -> Radio#%d:%d\n",
                id.c_str(), rc.first, rc.second);
    for (const auto& [rc, id] : framer_at)
        fprintf(stderr, "radio_difi:   framer   %s <- Radio#%d:%d\n",
                id.c_str(), rc.first, rc.second);
    for (const auto& [rc, id] : deframer_at)
        fprintf(stderr, "radio_difi:   deframer %s -> Radio#%d:%d\n",
                id.c_str(), rc.first, rc.second);
    const bool have_difi = !framer_at.empty();
    const bool have_defr = !deframer_at.empty();
    // rx=plain: receive from the ch2 (RF B:0) STOCK chain via divert (raw
    // sc16, no DIFI header) — the closed-loop partner for tx=difi, since
    // CAL_LOOPBACK only loops within one channel. tx=difi: transmit
    // DIFI-in-CHDR packets through the deframer chain (ep2 -> deframer ->
    // duc1 -> radio1:0 = ch2).
    const bool rx_plain = kv.count("rx") && kv["rx"] == "plain";
    d_tx_difi = kv.count("tx") && kv["tx"] == "difi";
    d_tx_sym  = kv.count("tx") && kv["tx"] == "sym";
    // tx=modcomp: packed plmod frame words through deframer -> plmod ->
    // rrc_interp -> DUC. Goes through the deframer (so it carries the DIFI
    // header the deframer strips), but the payload is uint32 words, not
    // samples — d_tx_raw switches send() to word (BE-store) mode.
    d_tx_modcomp = kv.count("tx") && kv["tx"] == "modcomp";
    d_tx_raw     = d_tx_modcomp;
    // tx=sc16: debug mode — same manual sc16 framing as tx=difi but with no
    // DIFI header, into the stock chain; isolates host framing from the
    // deframer FPGA block.
    const bool tx_sc16 = kv.count("tx") && kv["tx"] == "sc16";
    d_tx_hdrw = (d_tx_difi || d_tx_sym || d_tx_modcomp) ? DIFI_HDR_BYTES / 4 : 0;
    d_tx_manual = d_tx_difi || d_tx_sym || tx_sc16 || d_tx_modcomp;
    d_rx_hdr  = rx_plain ? 0 : DIFI_HDR_BYTES;
    // Debug overrides: pick any radio block / block-local channel for each
    // direction (defaults preserve the difi/plain semantics below).
    d_tx_chan = kv.count("tx_chan") ? std::stoi(kv["tx_chan"]) : 0;
    d_rx_chan = kv.count("rx_chan") ? std::stoi(kv["rx_chan"]) : 0;
    if (cfg.do_rx && !rx_plain && !have_difi) {
        fprintf(stderr, "radio_difi: no DIFI block in the image — is the "
                "radio flashed with rfnoc-difi?\n");
        return false;
    }
    if (cfg.do_tx && d_tx_sym && sym_at.empty()) {
        fprintf(stderr, "radio_difi: tx=sym but no symbol-TX chain "
                "(deframer -> rrc_interp -> DUC) in the image\n");
        return false;
    }
    if (cfg.do_tx && d_tx_sym && cfg.tx_rate <= 0.0) {
        fprintf(stderr, "radio_difi: tx=sym needs the app's symbol rate "
                "(--tx-symbols sets radio_config.tx_rate)\n");
        return false;
    }
    if (cfg.do_tx && d_tx_modcomp && modcomp_at.empty()) {
        fprintf(stderr, "radio_difi: tx=modcomp but no modulation-compressed "
                "chain (deframer -> plmod -> rrc_interp -> DUC) in the image\n");
        return false;
    }
    if (cfg.do_tx && d_tx_modcomp && cfg.tx_rate <= 0.0) {
        fprintf(stderr, "radio_difi: tx=modcomp needs the app's symbol rate "
                "(--tx-modcomp sets radio_config.tx_rate)\n");
        return false;
    }
    if (cfg.do_tx && d_tx_difi && !have_defr) {
        fprintf(stderr, "radio_difi: tx=difi but no difi_deframer (noc "
                "0xBD1F2) in the image\n");
        return false;
    }

    // RX radio: Radio#0 (DIFI framer chain, ch0) or Radio#1 ch0 (plain, ch2).
    // TX radio: Radio#0 (stock CHDR, ch0) or Radio#1 ch0 via the deframer.
    int rx_blk, tx_blk;
    if (!rx_plain) {
        // rx=difi: pick the framer chain — explicit rx_radio/rx_chan or the
        // first framer the image has.
        rx_blk = kv.count("rx_radio") ? std::stoi(kv["rx_radio"])
                                      : (have_difi ? framer_at.begin()->first.first : 0);
        if (!kv.count("rx_radio") && !kv.count("rx_chan") && have_difi)
            d_rx_chan = framer_at.begin()->first.second;
        if (have_difi) {
            auto it = framer_at.find({rx_blk, d_rx_chan});
            if (it == framer_at.end()) {
                fprintf(stderr, "radio_difi: no DIFI framer on Radio#%d:%d\n",
                        rx_blk, d_rx_chan);
                return false;
            }
            d_difi_id = uhd::rfnoc::block_id_t(it->second);
        }
    } else {
        rx_blk = kv.count("rx_radio") ? std::stoi(kv["rx_radio"]) : 1;
    }
    if (d_tx_modcomp) {
        tx_blk = kv.count("tx_radio") ? std::stoi(kv["tx_radio"])
                                      : modcomp_at.begin()->first.first;
        if (!kv.count("tx_radio") && !kv.count("tx_chan"))
            d_tx_chan = modcomp_at.begin()->first.second;
        auto it = modcomp_at.find({tx_blk, d_tx_chan});
        if (it == modcomp_at.end()) {
            fprintf(stderr, "radio_difi: no modcomp chain on Radio#%d:%d\n",
                    tx_blk, d_tx_chan);
            return false;
        }
        d_defr_id = uhd::rfnoc::block_id_t(it->second);
    } else if (d_tx_sym) {
        tx_blk = kv.count("tx_radio") ? std::stoi(kv["tx_radio"])
                                      : sym_at.begin()->first.first;
        if (!kv.count("tx_radio") && !kv.count("tx_chan"))
            d_tx_chan = sym_at.begin()->first.second;
        auto it = sym_at.find({tx_blk, d_tx_chan});
        if (it == sym_at.end()) {
            fprintf(stderr, "radio_difi: no symbol-TX chain on Radio#%d:%d\n",
                    tx_blk, d_tx_chan);
            return false;
        }
        d_defr_id = uhd::rfnoc::block_id_t(it->second);
    } else if (d_tx_difi) {
        tx_blk = kv.count("tx_radio") ? std::stoi(kv["tx_radio"])
                                      : (have_defr ? deframer_at.begin()->first.first : 0);
        if (!kv.count("tx_radio") && !kv.count("tx_chan") && have_defr)
            d_tx_chan = deframer_at.begin()->first.second;
        if (have_defr) {
            auto it = deframer_at.find({tx_blk, d_tx_chan});
            if (it == deframer_at.end()) {
                fprintf(stderr, "radio_difi: no DIFI deframer on Radio#%d:%d\n",
                        tx_blk, d_tx_chan);
                return false;
            }
            d_defr_id = uhd::rfnoc::block_id_t(it->second);
        }
    } else {
        tx_blk = kv.count("tx_radio") ? std::stoi(kv["tx_radio"]) : 0;
    }
    d_radio = d_graph->get_block<uhd::rfnoc::radio_control>(
        uhd::rfnoc::block_id_t(0, "Radio", rx_blk));
    d_radio_tx = tx_blk == rx_blk ? d_radio
        : d_graph->get_block<uhd::rfnoc::radio_control>(
              uhd::rfnoc::block_id_t(0, "Radio", tx_blk));
    const double radio_rate = d_radio->get_rate();
    fprintf(stderr, "radio_difi: radio rate %.3f Msps (image master rate)\n",
            radio_rate / 1e6);

    // TX path: through a DUC when the image has one (stock-faithful UC_200
    // core), else straight CHDR at the radio rate (early RF_BW-100 core).
    uhd::rfnoc::duc_block_control::sptr duc;
    if (cfg.do_tx) {
        try {
            duc = d_graph->get_block<uhd::rfnoc::duc_block_control>(
                uhd::rfnoc::block_id_t(0, "DUC", tx_blk));
        } catch (const std::exception&) { /* no DUC in image */ }
        if (!duc && std::abs(cfg.rate - radio_rate) > 1.0) {
            fprintf(stderr, "radio_difi: --rate %.3f Msps != radio rate %.3f "
                    "Msps and the image has no DUC; use --rate %.0f\n",
                    cfg.rate / 1e6, radio_rate / 1e6, radio_rate);
            return false;
        }
        d_radio_tx->set_tx_frequency(cfg.tx_freq, d_tx_chan);
        d_radio_tx->set_tx_gain(cfg.tx_gain, d_tx_chan);
        d_radio_tx->set_tx_antenna(tx_ant, d_tx_chan);
        if (duc) {
            // tx=sym: the host emits symbol-rate samples; the FPGA rrc_interp
            // interpolates 2x before the DUC, so the DUC ingests 2*tx_rate.
            const double duc_in = (d_tx_sym || d_tx_modcomp)
                                      ? 2.0 * cfg.tx_rate : cfg.rate;
            duc->set_output_rate(radio_rate, d_tx_chan);
            const double got = duc->set_input_rate(duc_in, d_tx_chan);
            fprintf(stderr, "radio_difi: DUC in %.3f Msps (req %.3f)%s\n",
                    got / 1e6, duc_in / 1e6,
                    (d_tx_sym || d_tx_modcomp)
                        ? " [symbol-rate: 2x rrc_interp ahead]" : "");
            if (std::abs(got - duc_in) > 1.0) {
                fprintf(stderr, "radio_difi: DUC cannot make %.3f from %.3f "
                        "Msps (integer interp only — pick sym_rate = %.2fM/"
                        "2^k)\n", duc_in / 1e6, radio_rate / 1e6,
                        radio_rate / 2e6);
                return false;
            }
        }
        fprintf(stderr, "radio_difi: TX Radio#%d:%d freq=%.3f MHz gain=%.1f dB "
                "ant=%s rate=%.3f Msps (%s%s)\n", tx_blk, d_tx_chan, cfg.tx_freq / 1e6,
                cfg.tx_gain, tx_ant.c_str(), cfg.rate / 1e6,
                d_tx_modcomp ? "MODULATION-compressed via deframer+plmod+rrc_interp"
                : d_tx_sym ? "SYMBOL-rate DIFI via deframer+rrc_interp"
                : d_tx_difi ? "DIFI-in-CHDR via deframer" : "CHDR",
                duc ? " + DUC" : "");
    }
    if (cfg.do_rx) {
        d_radio->set_rx_frequency(cfg.rx_freq, d_rx_chan);
        d_radio->set_rx_gain(cfg.rx_gain, d_rx_chan);
        d_radio->set_rx_antenna(rx_ant, d_rx_chan);
        try {
            d_ddc = d_graph->get_block<uhd::rfnoc::ddc_block_control>(
                uhd::rfnoc::block_id_t(0, "DDC", rx_blk));
            d_ddc->set_input_rate(radio_rate, d_rx_chan);
            const double got = d_ddc->set_output_rate(cfg.rate, d_rx_chan);
            fprintf(stderr, "radio_difi: DDC out %.3f Msps (req %.3f)\n",
                    got / 1e6, cfg.rate / 1e6);
        } catch (const std::exception& e) {
            fprintf(stderr, "radio_difi: no DDC configured (%s)\n", e.what());
        }
        fprintf(stderr, "radio_difi: RX Radio#%d:%d freq=%.3f MHz gain=%.1f dB "
                "ant=%s -> %s/UDP %s:%s\n", rx_blk, d_rx_chan, cfg.rx_freq / 1e6,
                cfg.rx_gain, rx_ant.c_str(), rx_plain ? "raw-sc16" : "DIFI",
                dest_addr.c_str(), dest_port.c_str());

        // Bind the DIFI sink socket BEFORE arming the stream.
        d_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (d_sock < 0) { perror("radio_difi: socket"); return false; }
        int rcvbuf = 64 << 20;
        setsockopt(d_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port   = htons((uint16_t)std::stoi(dest_port));
        if (inet_pton(AF_INET, dest_addr.c_str(), &sa.sin_addr) != 1 ||
            bind(d_sock, (sockaddr*)&sa, sizeof(sa)) < 0) {
            perror("radio_difi: bind (is dest_addr this host's data IP?)");
            return false;
        }

        // Remote raw-payload RX streamer: the FPGA sends the DIFI packets
        // straight to dest_addr:dest_port; no CHDR ever reaches the host.
        // (Arg names per UHD 4.8 remote streaming — see examples/python/
        // remote_rx.py; "addr"/"port" are silently ignored.)
        uhd::stream_args_t rargs("sc16", "sc16");
        rargs.args["dest_addr"]   = dest_addr;
        rargs.args["dest_port"]   = dest_port;
        if (!adapter.empty())     // optional; UHD routes by dest_addr if unset
            rargs.args["adapter"] = adapter;
        rargs.args["stream_mode"] =
            kv.count("stream_mode") ? kv["stream_mode"] : "raw_payload";
        if (kv.count("dest_mac_addr"))
            rargs.args["dest_mac_addr"] = kv["dest_mac_addr"];
        d_rx = d_graph->create_rx_streamer(1, rargs);
        // Attach at the chain's SEP-facing edge: framer (DIFI) or DDC (plain).
        d_graph->connect(rx_plain ? d_ddc->get_block_id() : d_difi_id,
                         rx_plain ? d_rx_chan : 0, d_rx, 0);
        d_issue_on_streamer = kv.count("issue") && kv["issue"] == "streamer";
    }

    if (cfg.do_tx) {
        // tx=difi sends raw bytes (28B DIFI header + sc16 samples) per packet,
        // so the streamer must not convert: cpu==wire==sc16. The plain CHDR
        // path keeps fc32 in / sc16 on the wire.
        uhd::stream_args_t targs(d_tx_manual ? "sc16" : "fc32", "sc16");
        targs.args["num_send_frames"] = std::to_string(cfg.uhd_num_send_frames);
        if (cfg.uhd_send_buff_size > 0)
            targs.args["send_buff_size"] = std::to_string(cfg.uhd_send_buff_size);
        d_tx = d_graph->create_tx_streamer(1, targs);
        // Enter the static TX chain at its SEP-facing edge (deframer for
        // tx=difi, else the DUC if present, else the radio).
        const bool via_defr = d_tx_difi || d_tx_sym || d_tx_modcomp;
        d_graph->connect(d_tx, 0,
                         via_defr ? d_defr_id
                                  : duc ? duc->get_block_id()
                                        : d_radio_tx->get_block_id(),
                         via_defr ? 0 : d_tx_chan);
        d_tx_spp = 0;   // resolved after commit (needs the MTU negotiated)
    }

    d_graph->commit();
    if (cfg.do_tx && d_tx_manual) {
        d_tx_spp = d_tx->get_max_num_samps();
        if (d_tx_sym) {
            // The rrc_interp block doubles the payload; keep the doubled
            // output packet under the CHDR MTU (2^10 words = 8 KB) or the
            // chain jams: input <= ~992 items -> output <= ~7.9 KB.
            // (tx=modcomp does NOT need this: the plmod consumes our words
            // and emits its own SPP_OUT=512 symbol packets, so the rrc only
            // ever doubles plmod-sized packets, never our input packet.)
            d_tx_spp = std::min<size_t>(d_tx_spp, 992);
        }
        if (kv.count("tx_spp")) {     // debug: force items/packet, verbatim
            d_tx_spp = std::min<size_t>(d_tx_spp, std::stoul(kv["tx_spp"]));
        } else if ((d_tx_spp - d_tx_hdrw) & 1) {
            // deframer v1 corrupted odd post-strip payloads (stock DUC pads
            // odd packets -> ~1-sample slip/pkt); v2 repacks any count, but
            // an even default costs nothing and works on both. Explicit
            // tx_spp bypasses this (e.g. to exercise the v2 residue path).
            d_tx_spp -= 1;
        }   // sc16 items per packet
        if (d_tx_spp <= d_tx_hdrw + 1) {
            fprintf(stderr, "radio_difi: TX spp %zu too small\n", d_tx_spp);
            return false;
        }
        d_tx_scratch.resize(d_tx_spp * 2);      // int16 I+Q per item
        fprintf(stderr, "radio_difi: TX manual sc16 framing: %zu items/packet "
                "(%zu payload samples + %zu header words)\n",
                d_tx_spp, d_tx_spp - d_tx_hdrw, d_tx_hdrw);
    }

    // Time: single device, no cross-device alignment needed — zero the
    // timekeeper and let the app schedule TX/RX starts against it.
    d_tk = d_graph->get_mb_controller(0)->get_timekeeper(0);
    d_tk->set_time_now(uhd::time_spec_t(0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    fprintf(stderr, "radio_difi: timekeeper zeroed (now=%.3f s)\n",
            get_time_now());
    return true;
}

void difi_radio_device::start_rx_continuous(double at_time_secs)
{
    uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    cmd.stream_now = false;
    cmd.time_spec  = uhd::time_spec_t(at_time_secs);
    /* Issue on the radio block: with a remote (raw_payload) streamer the
     * samples never reach the host streamer, and action forwarding from the
     * streamer edge may not resolve — the radio is the authoritative source
     * of the stream command either way. */
    if (d_issue_on_streamer) d_rx->issue_stream_cmd(cmd);
    else                     d_radio->issue_stream_cmd(cmd, d_rx_chan);
}

void difi_radio_device::stop_rx()
{
    uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    if (d_issue_on_streamer) d_rx->issue_stream_cmd(cmd);
    else                     d_radio->issue_stream_cmd(cmd, d_rx_chan);
}

size_t difi_radio_device::recv(void* buf, size_t nsamps, radio_rx_meta& md,
                               double timeout)
{
    md.error = radio_rx_error::none;

    pollfd pfd{d_sock, POLLIN, 0};
    int pr = poll(&pfd, 1, (int)(timeout * 1000));
    if (pr <= 0) {
        md.error = radio_rx_error::timeout;
        return 0;
    }
    ssize_t len = ::recv(d_sock, d_dgram, sizeof(d_dgram), 0);
    if (len <= (ssize_t)d_rx_hdr) {
        md.error = radio_rx_error::other;
        return 0;
    }

    if (d_rx_hdr == DIFI_HDR_BYTES) {
        /* v1.1 framers interleave Standard Flow Signal Context packets on
         * the same flow — keep signal-data packets only (type nibble 0x1). */
        if ((d_dgram[0] >> 4) != 0x1)
            return 0;
        /* Header word 0 carries a 4-bit per-stream sequence number (data
         * packets count separately from context per VITA 49); a gap means
         * the NIC/socket dropped datagrams — surface as overflow so the
         * app's UHD_OF counter tracks it. (rx=plain: no header, no seq.) */
        const int seq = d_dgram[1] & 0x0F;
        if (d_last_seq >= 0 && seq != ((d_last_seq + 1) & 0x0F))
            md.error = radio_rx_error::overflow;
        d_last_seq = seq;
    }

    const size_t have = ((size_t)len - d_rx_hdr) / 4;
    const size_t n    = have < nsamps ? have : nsamps;
    std::complex<float>* out = reinterpret_cast<std::complex<float>*>(buf);
    constexpr float k = 1.0f / 32768.0f;
    if (d_rx_hdr == DIFI_HDR_BYTES) {
        /* DIFI v1.1 payload: big-endian int16 (I,Q) pairs. */
        const uint8_t* b = d_dgram + d_rx_hdr;
        for (size_t i = 0; i < n; i++) {
            const int16_t re = (int16_t)((b[4 * i + 0] << 8) | b[4 * i + 1]);
            const int16_t im = (int16_t)((b[4 * i + 2] << 8) | b[4 * i + 3]);
            out[i] = d_iq_swapped ? std::complex<float>{ im * k, re * k }
                                  : std::complex<float>{ re * k, im * k };
        }
    } else {
        /* rx=plain (stock-chain divert): CHDR-native little-endian sc16. */
        const int16_t* in = reinterpret_cast<const int16_t*>(d_dgram + d_rx_hdr);
        for (size_t i = 0; i < n; i++)
            out[i] = d_iq_swapped
                ? std::complex<float>{ in[2 * i + 1] * k, in[2 * i + 0] * k }
                : std::complex<float>{ in[2 * i + 0] * k, in[2 * i + 1] * k };
    }
    return n;
}

size_t difi_radio_device::send(const void* buf, size_t nsamps,
                               const radio_tx_meta& m, double timeout)
{
    if (!d_tx_manual) {
        uhd::tx_metadata_t md;
        md.start_of_burst = m.start_of_burst;
        md.end_of_burst   = m.end_of_burst;
        md.has_time_spec  = m.has_time_spec;
        md.time_spec      = uhd::time_spec_t(m.time_spec);
        const void* p = buf ? buf : "";
        return d_tx->send(p, nsamps, md, timeout);
    }

    if (d_tx_raw) {
        /* Modulation-compressed TX: the app hands packed dvb_plmod frame
         * words (2 uint32 per complexf ring slot). Forward them verbatim
         * through the deframer — it strips our DIFI header and swab16's each
         * word, so big-endian-storing each word here makes the plmod see it
         * exactly (calibrated against the deframer's validated sc16 byte
         * order). The words are opaque to the deframer (treated as samples);
         * the plmod parses prologue+PLHEADER+bits and generates the symbols,
         * emitting its own SPP_OUT=512 packets downstream. Untimed stream. */
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
            h[1] = 0x60 | (d_tx_seq++ & 0x0F);
            const uint16_t words = (uint16_t)(n + HDRW);
            h[2] = (words >> 8) & 0xFF; h[3] = words & 0xFF;
            h[4] = 0x88; h[5] = 0x88; h[6] = 0x88; h[7] = 0x88;
            uint8_t* pb = reinterpret_cast<uint8_t*>(d_tx_scratch.data() + HDRW * 2);
            for (size_t i = 0; i < n; i++) {
                const uint32_t word = (buf && nsamps) ? w[done + i] : 0u;
                pb[4 * i + 0] = (uint8_t)(word >> 24);   // big-endian store
                pb[4 * i + 1] = (uint8_t)(word >> 16);
                pb[4 * i + 2] = (uint8_t)(word >> 8);
                pb[4 * i + 3] = (uint8_t)(word);
            }
            uhd::tx_metadata_t md;
            md.start_of_burst = m.start_of_burst && done == 0;
            md.end_of_burst   = m.end_of_burst && (done + n == total);
            md.has_time_spec  = false;     // modcomp streams untimed (like sym)
            const size_t sent =
                d_tx->send(d_tx_scratch.data(), n + HDRW, md, timeout);
            if (sent < n + HDRW) break;    // timeout/backpressure mid-fragment
            done += n;
        }
        return nsamps ? done / 2 : 0;      // report complexf ring slots consumed
    }

    /* DIFI-in-CHDR TX: one DIFI packet per CHDR packet (each send() call of
     * <= spp items emits exactly one packet; UHD never merges calls). Build
     * [7-word DIFI header][sc16 payload] per fragment; the FPGA deframer
     * strips the header and ignores its field values, so the header only
     * needs to be well-formed. The app hands fc32; convert here. */
    const size_t HDRW = d_tx_hdrw;                // 7 for difi, 0 for sc16
    const size_t payload_cap = d_tx_spp - HDRW;
    const auto* in = reinterpret_cast<const std::complex<float>*>(buf);
    // A 0-sample EOB still needs a packet the deframer can digest: 1 zero
    // sample after the header.
    const size_t total = nsamps ? nsamps : 1;
    size_t done = 0;
    while (done < total) {
        const size_t n = std::min(payload_cap, total - done);
        if (HDRW) {
            uint8_t* h = reinterpret_cast<uint8_t*>(d_tx_scratch.data());
            memset(h, 0, DIFI_HDR_BYTES);
            h[0] = 0x18;                              // DIFI data pkt w/ stream id
            h[1] = 0x60 | (d_tx_seq++ & 0x0F);        // TSI/TSF + 4-bit seq
            const uint16_t words = (uint16_t)(n + HDRW);
            h[2] = (words >> 8) & 0xFF; h[3] = words & 0xFF;
            h[4] = 0x88; h[5] = 0x88; h[6] = 0x88; h[7] = 0x88;  // stream id
        }
        /* DIFI v1.1 payload is big-endian int16 on the wire (the deframer
         * byte-swaps back to CHDR order after stripping the header). In the
         * tx=sc16 debug mode (no header, stock chain) stay CHDR-native LE. */
        uint8_t* pb = reinterpret_cast<uint8_t*>(d_tx_scratch.data() + HDRW * 2);
        for (size_t i = 0; i < n; i++) {
            const float re = (buf && nsamps) ? in[done + i].real() : 0.0f;
            const float im = (buf && nsamps) ? in[done + i].imag() : 0.0f;
            const int16_t ri = (int16_t)(re * 32767.0f);
            const int16_t qi = (int16_t)(im * 32767.0f);
            if (HDRW) {
                pb[4 * i + 0] = (uint8_t)(ri >> 8); pb[4 * i + 1] = (uint8_t)ri;
                pb[4 * i + 2] = (uint8_t)(qi >> 8); pb[4 * i + 3] = (uint8_t)qi;
            } else {
                int16_t* p = d_tx_scratch.data() + HDRW * 2;
                p[2 * i + 0] = ri;
                p[2 * i + 1] = qi;
            }
        }
        uhd::tx_metadata_t md;
        md.start_of_burst = m.start_of_burst && done == 0;
        md.end_of_burst   = m.end_of_burst && (done + n == total);
        /* tx=sym: stream untimed — the rrc_interp chain stalls on timed
         * bursts (and per the deframer contract, timestamps only stay
         * sample-accurate on pre-aligned packets anyway). */
        md.has_time_spec  = !d_tx_sym && m.has_time_spec && done == 0;
        md.time_spec      = uhd::time_spec_t(m.time_spec);
        const size_t sent =
            d_tx->send(d_tx_scratch.data(), n + HDRW, md, timeout);
        if (sent < n + HDRW) break;   // timeout/backpressure mid-fragment
        done += n;
    }
    return nsamps ? done : 0;
}

radio_async_msg difi_radio_device::poll_tx_async()
{
    radio_async_msg r;
    uhd::async_metadata_t a;
    if (d_tx && d_tx->recv_async_msg(a, 0.0)) {
        r.present    = true;
        r.underflow  = (a.event_code ==
                        uhd::async_metadata_t::EVENT_CODE_UNDERFLOW);
        r.event_code = (unsigned)a.event_code;
        r.time_spec  = a.time_spec.get_real_secs();
    }
    return r;
}

} // namespace

extern "C" radio_device* radio_device_create()
{
    return new difi_radio_device();
}

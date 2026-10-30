/* SPDX-License-Identifier: Apache-2.0
 * Ported from melkael/dvb-compressed-fh rx-compress (7227c9e).
 */
#include <cstdint>
#include <complex>
#include <functional>
#include <map>
#include <string>
#include <vector>

#pragma once

namespace difipack
{
  /*! A DIFI compression *framework* (candidate / experimental).
   *
   * The goal is not one compression scheme but a reusable way to carry ANY
   * compressed or alternate IQ representation inside standard DIFI/VITA-49.2
   * packets, in either direction (RX framer / TX deframer), dispatched by a
   * per-packet profile id (the VRT Packet Class Code, PCC). A profile is the
   * unit a third party plugs in; the DIFI transport (framer/deframer) is shared
   * and payload-blind.
   *
   * Profiles are organized in tiers by how DIFI-native they are:
   *
   *   Tier 0  NATIVE_VRT       Expressible in pure VITA-49.2: reduced item size
   *                            via the Data Packet Payload Format context field,
   *                            scale via the Reference Level / Gain context
   *                            field. No in-band header. A compression-unaware
   *                            VITA-49.2 receiver decodes it. (e.g. TRIM = sc8.)
   *   Tier 1  INBAND_ADAPTIVE  Native payload + a small in-band header carrying
   *                            per-packet-adaptive parameters (e.g. adaptive
   *                            shift, clip telemetry) that context packets are
   *                            too coarse/slow to convey.
   *   Tier 2  CODEC_EXTENSION  Something VRT cannot express at all: inter-sample
   *                            prediction (DPCM), or a *generative* payload
   *                            (TX modulation symbols/params -> synthesized
   *                            waveform). Carries a self-describing udCompHdr and
   *                            a candidate PCC; stock DIFI parses the framing
   *                            only.
   *
   * A stock DIFI receiver always parses the framing (Tier check is by PCC), and
   * fully decodes Tier 0. That graceful degradation is the point.
   */
  namespace compression
  {
    enum class Tier { NATIVE_VRT = 0, INBAND_ADAPTIVE = 1, CODEC_EXTENSION = 2 };
    enum class Direction { RX, TX, BOTH };

    // --- Candidate PCC registry (PROVISIONAL). Mirrors rfnoc_difi_regs.vh. ---
    // Tier-0 native profiles ride the *standard* data PCC and are told apart by
    // the payload-format context field, so they need no dedicated class code.
    static constexpr std::uint16_t PCC_STANDARD_DATA = 0x0000; // raw sc16 or Tier-0 native
    static constexpr std::uint16_t PCC_DPCM          = 0xDC02; // Tier-2 DPCM (RX)
    static constexpr std::uint16_t PCC_TX_MODULATION = 0xDC81; // Tier-2 generative (TX)

    //! Stream parameters recovered from the DIFI framing + standard context
    //! packet (NOT from any in-band header). This is what makes Tier-0 native.
    struct context_params
    {
      std::uint8_t item_bits  = 16; //!< Data Packet Payload Format: data item size
      int          scale_shift = 0; //!< scale carried by Reference Level / Gain
    };

    // ---- Tier-0 native decode: packed sc8/sc16 -> sc16, scale from context ----
    std::vector<std::complex<std::int16_t>>
    decode_native(const std::vector<std::uint8_t> &payload,
                  const context_params &ctx, std::uint16_t n);

    // ---- Tier-2 DPCM (self-describing in-band udCompHdr) ----
    enum class DpcmMode : std::uint8_t { DPCM = 0xC2 };
    struct dpcm_header
    {
      std::uint8_t  r_bits;
      std::uint8_t  shift;
      std::uint16_t n;
    };
    dpcm_header parse_dpcm_header(const std::vector<std::uint8_t> &payload);
    std::vector<std::complex<std::int16_t>>
    decode_dpcm(const std::vector<std::uint8_t> &payload);

    // ---- Reference encoders (mirror the FPGA iq_compress RTL) ----
    //! Tier-0 native trim: sc16 -> packed sc8, NO in-band header.
    std::vector<std::uint8_t>
    encode_trim_native(const std::vector<std::complex<std::int16_t>> &samples,
                       std::uint8_t shift);
    //! Tier-2 DPCM: sc16 -> udCompHdr + packed residuals.
    std::vector<std::uint8_t>
    encode_dpcm(const std::vector<std::complex<std::int16_t>> &samples,
                std::uint8_t shift, std::uint8_t r_bits);

    // ---- Tier-2 generative TX profile (modulation) ----
    //! The DIFI deframer restores DIFI big-endian sc16 to CHDR-native order by
    //! swapping the bytes within each int16 of every payload word. A generative
    //! TX payload (plmod frame words) is not sc16, so to stay DIFI-native and
    //! reuse the stock deframer unchanged, the generative profile DEFINES its
    //! payload as carried in that same DIFI byte order: the host pre-applies
    //! this swap per word so the deframer's swap cancels and plmod receives the
    //! native frame words. `wire_swab16` is an involution (its own inverse).
    inline std::uint32_t wire_swab16(std::uint32_t w)
    {
      return ((w >> 8) & 0x00FF00FFu) | ((w << 8) & 0xFF00FF00u);
    }

    //! Encode a plmod frame (native 32-bit words) into the DIFI payload bytes
    //! for PCC_TX_MODULATION: pre-swapped words, big-endian on the wire.
    std::vector<std::uint8_t>
    encode_tx_modulation(const std::vector<std::uint32_t> &frame_words);

    //! Inverse of the deframer's payload transform: recover the native frame
    //! words a plmod block receives from a PCC_TX_MODULATION DIFI payload.
    //! (Models `difi_deframer` for host-side round-trip tests.)
    std::vector<std::uint32_t>
    deframe_tx_modulation(const std::vector<std::uint8_t> &payload);

    // ---- Minimal DIFI prologue view (difipack is encode-only) for the
    //      compression-unaware "stock" framing parse (interop check) ----
    struct prologue_view
    {
      std::uint8_t  packet_type;
      std::uint8_t  tsi;
      std::uint8_t  seqnum;
      std::uint16_t packet_size;
      std::uint32_t stream_id;
      std::uint32_t oui;
      std::uint16_t icc;
      std::uint16_t pcc;
      std::uint32_t int_timestamp;
      std::uint64_t frac_timestamp;
    };
    prologue_view parse_prologue(const std::vector<std::uint8_t> &packet_bytes);

    // ---- The profile registry: the plug-in point ----
    //! A registered profile. `decode` maps a received VRT payload (+ context)
    //! to sc16 samples. TX/generative profiles that a host cannot expand to IQ
    //! leave `decode` empty (host_decodable == false).
    struct profile
    {
      std::string  name;
      std::uint16_t pcc;
      Tier         tier;
      Direction    direction;
      bool         host_decodable; //!< false for generative TX profiles
      std::function<std::vector<std::complex<std::int16_t>>(
          const std::vector<std::uint8_t> &payload,
          const context_params &ctx, std::uint16_t n)>
          decode;
    };

    //! The built-in registry (TRIM native, DPCM, TX modulation). Keyed by PCC.
    //! For PCC_STANDARD_DATA the payload format (ctx.item_bits) selects raw sc16
    //! vs Tier-0 native trim.
    const std::map<std::uint16_t, profile> &registry();

    //! Look up the profile advertised by a parsed prologue (+ context), or
    //! nullptr if the PCC is unknown (a stock receiver would still parse framing).
    const profile *resolve(const prologue_view &p, const context_params &ctx);
  }
}

namespace difipack::compression {
// Allocation-free RX expansion, shared with the RFNoC plugin.
std::size_t decode_fc32(const std::uint8_t* payload, std::size_t bytes,
    std::complex<float>* out, std::size_t max_samples, int mode, int scale_shift);
}

namespace difipack::compression {
// CPU sc16 input whose UHD I/Q-to-item32 packing followed by the FPGA
// deframer's byte swap produces the requested native modulation word.
inline std::complex<std::int16_t> uhd_input_word(std::uint32_t native_word)
{
    const auto before_deframer = wire_swab16(native_word);
    return {static_cast<std::int16_t>(before_deframer >> 16),
            static_cast<std::int16_t>(before_deframer & 0xFFFF)};
}
}

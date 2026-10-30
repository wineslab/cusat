/* SPDX-License-Identifier: Apache-2.0
 * Ported from melkael/dvb-compressed-fh rx-compress (7227c9e).
 */
#include "trx/difi_compression.h"

#include <algorithm>
#include <stdexcept>

namespace
{
  std::int8_t clip8(int v)
  {
    if (v > 127)  return 127;
    if (v < -127) return -127;
    return (std::int8_t)v;
  }

  std::int8_t trim_comp(std::int16_t v, std::uint8_t sh)
  {
    int r = (sh == 0) ? v : ((v + (1 << (sh - 1))) >> sh);
    return clip8(r);
  }

  std::uint32_t read_be32(const std::vector<std::uint8_t> &b, std::size_t off)
  {
    return ((std::uint32_t)b[off] << 24) | ((std::uint32_t)b[off + 1] << 16) |
           ((std::uint32_t)b[off + 2] << 8) | ((std::uint32_t)b[off + 3]);
  }

  std::int16_t read_be16(const std::vector<std::uint8_t> &b, std::size_t off)
  {
    return (std::int16_t)(((std::uint16_t)b[off] << 8) | b[off + 1]);
  }

  std::int32_t sign_extend(std::uint32_t u, int n)
  {
    std::uint32_t sign = 1u << (n - 1);
    if (u & sign)
      return (std::int32_t)(u | ~((1u << n) - 1));
    return (std::int32_t)u;
  }

  // MSB-first bit reader over a byte range.
  class bit_reader
  {
  public:
    bit_reader(const std::vector<std::uint8_t> &b, std::size_t off)
        : _b(b), _byte(off), _bit(0) {}
    std::uint32_t get(int n)
    {
      std::uint32_t v = 0;
      for (int i = 0; i < n; i++)
      {
        std::uint32_t bit =
            (_byte < _b.size()) ? ((_b[_byte] >> (7 - _bit)) & 1u) : 0u;
        v = (v << 1) | bit;
        if (++_bit == 8) { _bit = 0; _byte++; }
      }
      return v;
    }
  private:
    const std::vector<std::uint8_t> &_b;
    std::size_t _byte;
    int _bit;
  };
}

namespace difipack
{
  namespace compression
  {
    std::vector<std::complex<std::int16_t>>
    decode_native(const std::vector<std::uint8_t> &payload,
                  const context_params &ctx, std::uint16_t n)
    {
      std::vector<std::complex<std::int16_t>> out;
      out.reserve(n);
      const int sh = ctx.scale_shift;
      if (sh < 0 || sh > 8) throw std::invalid_argument("scale shift out of range");
      if (payload.size() < (size_t)n * (ctx.item_bits / 4))
        throw std::invalid_argument("truncated native payload");
      if (ctx.item_bits == 8)
      {
        // Tier-0 trim: packed sc8 (2 samples/word), no in-band header.
        for (std::uint16_t i = 0; i < n; i++)
        {
          std::int8_t ci = (std::int8_t)payload.at(2 * i);
          std::int8_t cq = (std::int8_t)payload.at(2 * i + 1);
          out.emplace_back((std::int16_t)(ci * (1 << sh)), (std::int16_t)(cq * (1 << sh)));
        }
      }
      else if (ctx.item_bits == 16)
      {
        // Raw sc16 (uncompressed reference); scale is normally 0 here.
        for (std::uint16_t i = 0; i < n; i++)
          out.emplace_back((std::int16_t)(read_be16(payload, 4 * i) * (1 << sh)),
                           (std::int16_t)(read_be16(payload, 4 * i + 2) * (1 << sh)));
      }
      else
        throw std::invalid_argument("unsupported native item_bits");
      return out;
    }

    dpcm_header parse_dpcm_header(const std::vector<std::uint8_t> &payload)
    {
      if (payload.size() < 4)
        throw std::invalid_argument("payload too short for udCompHdr");
      std::uint32_t w = read_be32(payload, 0);
      if (((w >> 24) & 0xFF) != (std::uint8_t)DpcmMode::DPCM)
        throw std::invalid_argument("not a DPCM udCompHdr");
      dpcm_header h;
      h.r_bits = (std::uint8_t)((w >> 20) & 0xF);
      h.shift  = (std::uint8_t)((w >> 16) & 0xF);
      h.n      = (std::uint16_t)(w & 0xFFFF);
      return h;
    }

    std::vector<std::complex<std::int16_t>>
    decode_dpcm(const std::vector<std::uint8_t> &payload)
    {
      dpcm_header h = parse_dpcm_header(payload);
      if (h.r_bits == 0 || h.r_bits > 8)
        throw std::invalid_argument("DPCM R out of range");
      if (h.shift > 8 || h.n == 0 || (payload.size() - 4) * 8 < 16u + 2u * h.r_bits * (h.n - 1))
        throw std::invalid_argument("invalid or truncated DPCM payload");
      std::vector<std::complex<std::int16_t>> out;
      out.reserve(h.n);
      bit_reader br(payload, 4);
      std::int32_t ri = sign_extend(br.get(8), 8);
      std::int32_t rq = sign_extend(br.get(8), 8);
      out.emplace_back((std::int16_t)(ri * (1 << h.shift)), (std::int16_t)(rq * (1 << h.shift)));
      for (std::uint16_t i = 1; i < h.n; i++)
      {
        ri = clip8(ri + sign_extend(br.get(h.r_bits), h.r_bits));
        rq = clip8(rq + sign_extend(br.get(h.r_bits), h.r_bits));
        out.emplace_back((std::int16_t)(ri * (1 << h.shift)), (std::int16_t)(rq * (1 << h.shift)));
      }
      return out;
    }

    std::vector<std::uint8_t>
    encode_trim_native(const std::vector<std::complex<std::int16_t>> &samples,
                       std::uint8_t shift)
    {
      if (shift > 8) throw std::invalid_argument("trim shift out of range");
      std::vector<std::uint8_t> p;
      for (const auto &s : samples)
      {
        p.push_back((std::uint8_t)trim_comp(s.real(), shift));
        p.push_back((std::uint8_t)trim_comp(s.imag(), shift));
      }
      while (p.size() % 4 != 0) p.push_back(0); // pad odd count to a whole word
      return p;
    }

    std::vector<std::uint8_t>
    encode_dpcm(const std::vector<std::complex<std::int16_t>> &samples,
                std::uint8_t shift, std::uint8_t r_bits)
    {
      if (samples.empty() || samples.size() > 65535 || shift > 8 || r_bits < 1 || r_bits > 8)
        throw std::invalid_argument("invalid DPCM parameters");
      std::uint16_t n = (std::uint16_t)samples.size();
      std::vector<std::uint8_t> p;
      std::uint32_t hdr = ((std::uint32_t)DpcmMode::DPCM << 24) |
                          ((std::uint32_t)(r_bits & 0xF) << 20) |
                          ((std::uint32_t)(shift & 0xF) << 16) | n;
      p.push_back((hdr >> 24) & 0xFF); p.push_back((hdr >> 16) & 0xFF);
      p.push_back((hdr >> 8) & 0xFF);  p.push_back(hdr & 0xFF);

      std::uint64_t acc = 0; int nbits = 0;
      auto push_bits = [&](std::uint32_t val, int w) {
        for (int i = w - 1; i >= 0; i--)
        {
          acc = (acc << 1) | ((val >> i) & 1u);
          if (++nbits == 8) { p.push_back((std::uint8_t)acc); acc = 0; nbits = 0; }
        }
      };
      const int rmax = (1 << (r_bits - 1)) - 1, rmin = -(1 << (r_bits - 1));
      std::int32_t ri = trim_comp(samples[0].real(), shift);
      std::int32_t rq = trim_comp(samples[0].imag(), shift);
      push_bits((std::uint8_t)ri, 8);
      push_bits((std::uint8_t)rq, 8);
      for (std::uint16_t i = 1; i < n; i++)
      {
        int ti = trim_comp(samples[i].real(), shift);
        int tq = trim_comp(samples[i].imag(), shift);
        int ei = std::max(rmin, std::min(rmax, ti - ri));
        int eq = std::max(rmin, std::min(rmax, tq - rq));
        ri = clip8(ri + ei);
        rq = clip8(rq + eq);
        push_bits((std::uint32_t)ei & ((1u << r_bits) - 1), r_bits);
        push_bits((std::uint32_t)eq & ((1u << r_bits) - 1), r_bits);
      }
      if (nbits > 0) p.push_back((std::uint8_t)(acc << (8 - nbits)));
      while (p.size() % 4 != 0) p.push_back(0);
      return p;
    }

    std::vector<std::uint8_t>
    encode_tx_modulation(const std::vector<std::uint32_t> &frame_words)
    {
      // Pre-swap each word so the deframer's swab16 cancels; serialize each
      // resulting word big-endian (DIFI wire order).
      std::vector<std::uint8_t> p;
      p.reserve(frame_words.size() * 4);
      for (std::uint32_t w : frame_words)
      {
        std::uint32_t s = wire_swab16(w);
        p.push_back((s >> 24) & 0xFF); p.push_back((s >> 16) & 0xFF);
        p.push_back((s >> 8) & 0xFF);  p.push_back(s & 0xFF);
      }
      return p;
    }

    std::vector<std::uint32_t>
    deframe_tx_modulation(const std::vector<std::uint8_t> &payload)
    {
      // Read each big-endian word, apply swab16 (as the deframer does) ->
      // native frame words that plmod consumes.
      std::vector<std::uint32_t> out;
      out.reserve(payload.size() / 4);
      for (std::size_t i = 0; i + 4 <= payload.size(); i += 4)
        out.push_back(wire_swab16(read_be32(payload, i)));
      return out;
    }

    prologue_view parse_prologue(const std::vector<std::uint8_t> &b)
    {
      if (b.size() < 28)
        throw std::invalid_argument("packet too short for DIFI prologue");
      std::uint32_t w0 = read_be32(b, 0);
      prologue_view v;
      v.packet_type    = (std::uint8_t)((w0 >> 28) & 0xF);
      v.tsi            = (std::uint8_t)((w0 >> 22) & 0x3);
      v.seqnum         = (std::uint8_t)((w0 >> 16) & 0xF);
      v.packet_size    = (std::uint16_t)(w0 & 0xFFFF);
      v.stream_id      = read_be32(b, 4);
      v.oui            = read_be32(b, 8) & 0x00FFFFFF;
      std::uint32_t w3 = read_be32(b, 12);
      v.icc            = (std::uint16_t)(w3 >> 16);
      v.pcc            = (std::uint16_t)(w3 & 0xFFFF);
      v.int_timestamp  = read_be32(b, 16);
      v.frac_timestamp = ((std::uint64_t)read_be32(b, 20) << 32) | read_be32(b, 24);
      return v;
    }

    const std::map<std::uint16_t, profile> &registry()
    {
      static const std::map<std::uint16_t, profile> r = {
          {PCC_STANDARD_DATA,
           {"trim/raw (native VRT)", PCC_STANDARD_DATA, Tier::NATIVE_VRT,
            Direction::RX, true,
            [](const std::vector<std::uint8_t> &pl, const context_params &ctx,
               std::uint16_t n) { return decode_native(pl, ctx, n); }}},
          {PCC_DPCM,
           {"DPCM", PCC_DPCM, Tier::CODEC_EXTENSION, Direction::RX, true,
            [](const std::vector<std::uint8_t> &pl, const context_params &,
               std::uint16_t) { return decode_dpcm(pl); }}},
          {PCC_TX_MODULATION,
           // Generative TX profile: the payload is modulation symbols/params
           // that the FPGA plmod expands into a waveform. Not host-decodable to
           // IQ without the modulator, so `decode` is empty -- it exists in the
           // registry so the same DIFI transport + PCC dispatch covers TX.
           {"TX modulation (generative)", PCC_TX_MODULATION,
            Tier::CODEC_EXTENSION, Direction::TX, false, nullptr}},
      };
      return r;
    }

    const profile *resolve(const prologue_view &p, const context_params &)
    {
      const auto &r = registry();
      auto it = r.find(p.pcc);
      return (it == r.end()) ? nullptr : &it->second;
    }
  }
}

std::size_t difipack::compression::decode_fc32(const std::uint8_t* p, std::size_t payload,
    std::complex<float>* out, std::size_t nsamps, int mode, int scale_shift)
{
    if (mode < 0 || mode > 2 || scale_shift < 0 || scale_shift > 8)
        throw std::invalid_argument("invalid compression configuration");
    if (mode == 0 && payload % 4) throw std::invalid_argument("unaligned sc16 payload");
    if (mode == 1 && payload % 2) throw std::invalid_argument("unaligned sc8 payload");
    constexpr float k = 1.0f / 32768.0f;

    if (mode == 0) {
        // Passthrough: DIFI-native big-endian sc16 (I,Q) pairs.
        const size_t have = payload / 4;
        const size_t n    = std::min(have, nsamps);
        for (size_t i = 0; i < n; i++) {
            const int16_t re = (int16_t)((p[4 * i + 0] << 8) | p[4 * i + 1]);
            const int16_t im = (int16_t)((p[4 * i + 2] << 8) | p[4 * i + 3]);
            out[i] = { re * k, im * k };
        }
        return n;
    }

    if (mode == 1) {
        // Trim (Tier-0 native): packed sc8 (I,Q) bytes, no in-band header.
        // Width & scale are context; SHIFT is host-known (set at setup). N is
        // context too — decode all sc8 pairs the payload carries (RX spp is
        // even, so there is no odd-count zero-pad word to trim).
        const float sc  = (float)(1 << scale_shift) / 32768.0f;
        const size_t have = payload / 2;
        const size_t n    = std::min(have, nsamps);
        for (size_t i = 0; i < n; i++) {
            const int8_t I = (int8_t)p[2 * i + 0];
            const int8_t Q = (int8_t)p[2 * i + 1];
            out[i] = { I * sc, Q * sc };
        }
        return n;
    }

    // DPCM (Tier-2): self-describing udCompHdr word 0 (big-endian):
    //   [31:24] 0xC2 | [23:20] R | [19:16] shift | [15:0] N
    // then an 8+8-bit sc8 anchor and 2*R residual bits per subsequent sample,
    // MSB-first over the big-endian payload byte stream; reconstruct in sc8.
    if (payload < 4) throw std::invalid_argument("short DPCM header");
    const uint32_t hdr = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                         ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
    if ((hdr >> 24) != 0xC2) throw std::invalid_argument("bad DPCM magic");
    const int R     = (int)((hdr >> 20) & 0xF);
    const int shift = (int)((hdr >> 16) & 0xF);
    const size_t N  = hdr & 0xFFFF;
    if (R == 0 || R > 8 || shift > 8 || N == 0) throw std::invalid_argument("bad DPCM parameters");
    const float sc = (float)(1 << shift) / 32768.0f;
    const size_t avail_bits = (payload - 4) * 8;           // bits after udCompHdr
    if (16u + 2u * (size_t)R * (N - 1) > avail_bits) throw std::invalid_argument("truncated DPCM payload");
    size_t n = std::min(N, nsamps);
    if (n && 16u + 2u * (size_t)R * (n - 1) > avail_bits)  // truncated pkt
        n = avail_bits >= 16 ? 1 + (avail_bits - 16) / (2u * R) : 0;
    if (n == 0) return 0;
    const uint32_t rmask = (1u << R) - 1u, rsign = 1u << (R - 1);
    // MSB-first bit accumulator over the big-endian byte stream (bytes are
    // already in wire order, so refill is a plain shift-in — no per-word swap).
    uint64_t acc = 0; int accn = 0; size_t bidx = 4;       // start past udCompHdr
    const uint8_t* pb = p;
#define COMPFH_REFILL(need)                                                 \
    while (accn < (int)(need)) {                                            \
        acc = (acc << 8) | pb[bidx++];                                      \
        accn += 8;                                                          \
    }
    COMPFH_REFILL(16);                                     // anchor sc8 I,Q
    accn -= 8; int ri = (int8_t)((acc >> accn) & 0xFFu);
    accn -= 8; int rq = (int8_t)((acc >> accn) & 0xFFu);
    out[0] = { ri * sc, rq * sc };
    for (size_t j = 1; j < n; j++) {
        COMPFH_REFILL(2 * R);
        accn -= R; int ei = (int)(((acc >> accn) & rmask) ^ rsign) - (int)rsign;
        accn -= R; int eq = (int)(((acc >> accn) & rmask) ^ rsign) - (int)rsign;
        ri += ei; ri = ri < -127 ? -127 : (ri > 127 ? 127 : ri);   // clip8
        rq += eq; rq = rq < -127 ? -127 : (rq > 127 ? 127 : rq);
        out[j] = { ri * sc, rq * sc };
    }
#undef COMPFH_REFILL
    return n;
}

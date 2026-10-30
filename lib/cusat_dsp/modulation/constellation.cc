/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Every builder fills points[label] for one family. Ring radii follow from
 * the ring ratios of the code rate and unit average energy; angles are in
 * units of pi. The tables list, per label row, the ring and the angles of
 * the quadrant variants selected by the remaining label bits.
 */

#include "modulation/constellation.h"
#include "modulation/apsk256_points.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cusat_dsp {

namespace {

using points_t = std::array<complexf, 256>;

complexf polar_point(double radius, double angle)
{
    return complexf((float)(radius * std::cos(angle)), (float)(radius * std::sin(angle)));
}

/* ---- DVB-S2 (EN 302 307-1 clause 5.4) -------------------------------- */

void build_qpsk(points_t& p)
{
    // bit 1 (MSB) keys the I sign, bit 0 the Q sign
    const float a = (float)M_SQRT1_2;
    for (int label = 0; label < 4; label++)
        p[label] = complexf((label & 2) ? -a : a, (label & 1) ? -a : a);
}

void build_8psk(points_t& p)
{
    const float a = (float)M_SQRT1_2;
    // figure 10: label -> point
    const complexf pts[8] = { { a, a }, { 1, 0 },   { -1, 0 }, { -a, -a },
                              { 0, 1 }, { a, -a },  { -a, a }, { 0, -1 } };
    for (int label = 0; label < 8; label++)
        p[label] = pts[label];
}

/* S2 16APSK and 32APSK are computed in single precision, angle k pi / d. */
struct s2_point_t {
    int ring;
    int k;
    int d;
};

complexf s2_point(float radius, int k, int d)
{
    const float pi = M_PI;
    const float angle = (float)(k * (double)pi / d);
    return complexf(radius * std::cos(angle), radius * std::sin(angle));
}

float s2_16apsk_gamma(code_rate_t rate)
{
    switch (rate) {
    case RATE_2_3: return 3.15;
    case RATE_3_4: return 2.85;
    case RATE_4_5: return 2.75;
    case RATE_5_6: return 2.70;
    case RATE_8_9: return 2.60;
    case RATE_9_10: return 2.57;
    case RATE_26_45: return 3.7;
    case RATE_3_5: return 3.7;
    case RATE_28_45: return 3.5;
    case RATE_23_36: return 3.1;
    case RATE_25_36: return 3.1;
    case RATE_13_18: return 2.85;
    case RATE_140_180: return 3.6;
    case RATE_154_180: return 3.2;
    case RATE_7_15: return 3.32;
    case RATE_8_15: return 3.50;
    default: return 2.85;
    }
}

void build_16apsk_4_12(points_t& p, code_rate_t rate)
{
    // 4+12APSK, figure 11 (also the S2X 4+12 rates of EN 302 307-2 Table 11a)
    static const s2_point_t LABELS[16] = {
        { 1, 1, 4 },   { 1, -1, 4 },  { 1, 3, 4 },   { 1, -3, 4 },
        { 1, 1, 12 },  { 1, -1, 12 }, { 1, 11, 12 }, { 1, -11, 12 },
        { 1, 5, 12 },  { 1, -5, 12 }, { 1, 7, 12 },  { 1, -7, 12 },
        { 0, 1, 4 },   { 0, -1, 4 },  { 0, 3, 4 },   { 0, -3, 4 },
    };
    const float gamma = s2_16apsk_gamma(rate);
    const float r1 = std::sqrt(16.0 / (4.0 + 12.0 * gamma * gamma));
    const float radius[2] = { r1, gamma * r1 };
    for (int label = 0; label < 16; label++)
        p[label] = s2_point(radius[LABELS[label].ring], LABELS[label].k, LABELS[label].d);
}

void build_32apsk_4_12_16(points_t& p, code_rate_t rate)
{
    // 4+12+16APSK, figure 12. Ring ratios g1 = R2/R1, g2 = R3/R1 of
    // EN 302 307-1 Table 10; unit average energy (clause 5.4.4):
    //     (4 R1^2 + 12 R2^2 + 16 R3^2) / 32 = 1
    //  => R1 = sqrt(8 / (1 + 3 g1^2 + 4 g2^2)).
    // Table 10 lists rates 3/4 .. 9/10; other rates reaching this builder
    // (none in cuSAT's MODCOD table) use the 3/4 ratios.
    double g1 = 2.84, g2 = 5.27; // 3/4
    switch (rate) {
    case RATE_4_5: g1 = 2.72; g2 = 4.87; break;
    case RATE_5_6: g1 = 2.64; g2 = 4.64; break;
    case RATE_8_9: g1 = 2.54; g2 = 4.33; break;
    case RATE_9_10: g1 = 2.53; g2 = 4.30; break;
    default: break;
    }
    const double r1 = std::sqrt(8.0 / (1.0 + 3.0 * g1 * g1 + 4.0 * g2 * g2));
    const float radius[3] = { (float)r1, (float)(g1 * r1), (float)(g2 * r1) };

    static const s2_point_t LABELS[32] = {
        { 1, 1, 4 },   { 1, 5, 12 },   { 1, -1, 4 },  { 1, -5, 12 },
        { 1, 3, 4 },   { 1, 7, 12 },   { 1, -3, 4 },  { 1, -7, 12 },
        { 2, 1, 8 },   { 2, 3, 8 },    { 2, -1, 4 },  { 2, -1, 2 },
        { 2, 3, 4 },   { 2, 1, 2 },    { 2, -7, 8 },  { 2, -5, 8 },
        { 1, 1, 12 },  { 0, 1, 4 },    { 1, -1, 12 }, { 0, -1, 4 },
        { 1, 11, 12 }, { 0, 3, 4 },    { 1, -11, 12 }, { 0, -3, 4 },
        { 2, 0, 1 },   { 2, 1, 4 },    { 2, -1, 8 },  { 2, -3, 8 },
        { 2, 7, 8 },   { 2, 5, 8 },    { 2, 1, 1 },   { 2, -3, 4 },
    };
    for (int label = 0; label < 32; label++)
        p[label] = s2_point(radius[LABELS[label].ring], LABELS[label].k, LABELS[label].d);
}

/* ---- DVB-S2X (EN 302 307-2 clause 5.4) ------------------------------- */

void build_8apsk(points_t& p, code_rate_t rate)
{
    // 2+4+2APSK, Table 10a/10b
    const double g1 = (rate == RATE_100_180) ? 5.32 : 6.39;
    const double g2 = (rate == RATE_100_180) ? 6.8 : 8.0;
    const double r1 = std::sqrt(8.0 / (2.0 + 4.0 * g1 * g1 + 2.0 * g2 * g2));
    const double radius[8] = { r1, g1 * r1, g1 * r1, g2 * r1, r1, g1 * r1, g1 * r1, g2 * r1 };
    const double angle[8] = { 1.0, 1.0 + 0.352, 1.0 - 0.352, 1.0, 0.0, -0.352, 0.352, 0.0 };
    for (int label = 0; label < 8; label++)
        p[label] = polar_point(radius[label], angle[label] * M_PI);
}

void build_16apsk_8_8(points_t& p, code_rate_t rate)
{
    // Explicit coordinates (Table 11e) for 18/30 and 20/30
    static const double COORD_18_30[16][2] = {
    { 0.4718,  0.2606}, { 0.2606,  0.4718}, {-0.4718,  0.2606}, {-0.2606,  0.4718},
    { 0.4718, -0.2606}, { 0.2606, -0.4718}, {-0.4718, -0.2606}, {-0.2606, -0.4718},
    { 1.2088,  0.4984}, { 0.4984,  1.2088}, {-1.2088,  0.4984}, {-0.4984,  1.2088},
    { 1.2088, -0.4984}, { 0.4984, -1.2088}, {-1.2088, -0.4984}, {-0.4984, -1.2088},
    };
    static const double COORD_20_30[16][2] = {
    { 0.5061,  0.2474}, { 0.2474,  0.5061}, {-0.5061,  0.2474}, {-0.2474,  0.5061},
    { 0.5061, -0.2474}, { 0.2474, -0.5061}, {-0.5061, -0.2474}, {-0.2474, -0.5061},
    { 1.2007,  0.4909}, { 0.4909,  1.2007}, {-1.2007,  0.4909}, {-0.4909,  1.2007},
    { 1.2007, -0.4909}, { 0.4909, -1.2007}, {-1.2007, -0.4909}, {-0.4909, -1.2007},
    };
    const double(*coord)[2] =
        (rate == RATE_18_30) ? COORD_18_30 : (rate == RATE_20_30) ? COORD_20_30 : nullptr;
    if (coord) {
        double energy = 0.0;
        for (int i = 0; i < 16; i++)
            energy += coord[i][0] * coord[i][0] + coord[i][1] * coord[i][1];
        const double scale = std::sqrt(16.0 / energy);
        for (int i = 0; i < 16; i++)
            p[i] = complexf((float)(coord[i][0] * scale), (float)(coord[i][1] * scale));
        return;
    }
    // Ring form (Tables 11c/11d): gamma = 2.19, angle n pi / 8 by (b0, p, q)
    const double gamma = 2.19;
    const double r1 = std::sqrt(2.0 / (1.0 + gamma * gamma));
    static const int N[2][2][2] = { { { 1, 15 }, { 7, 9 } }, { { 3, 13 }, { 5, 11 } } };
    for (int label = 0; label < 16; label++) {
        const int ring = (label >> 3) & 1, q = (label >> 2) & 1, pp = (label >> 1) & 1, b0 = label & 1;
        p[label] = polar_point(ring ? gamma * r1 : r1, N[b0][pp][q] * (M_PI / 8.0));
    }
}

/* Row-based APSK tables: a label splits into a row index and a quadrant
 * column (p, q); every row sits on one ring. */
struct ring_row_t {
    int ring;
    double angle[4]; // units of pi, columns (p, q) = 00, 01, 10, 11
};

void build_32apsk_rb(points_t& p, code_rate_t rate, fecframe_size_t framesize)
{
    // 4+12+16rbAPSK, Tables 12a/12b; row = (b3, b2, b0), column = (b4, b1)
    double g1 = 2.85, g2 = 5.55;
    if (rate == RATE_32_45) {
        g1 = 2.839;
        g2 = 5.258;
    } else if (framesize == FECFRAME_SHORT) {
        g1 = 2.84;
        g2 = 5.54;
    }
    const double r1 = std::sqrt(32.0 / (4.0 + 12.0 * g1 * g1 + 16.0 * g2 * g2));
    const double radius[4] = { 0.0, r1, g1 * r1, g2 * r1 };
    static const int RING[8] = { 3, 3, 2, 2, 3, 3, 2, 1 };
    static const double ANGLE[8][4] = {
    { 11.0/16,  5.0/16, 21.0/16, 27.0/16 }, // p00q0
    {  9.0/16,  7.0/16, 23.0/16, 25.0/16 }, // p00q1
    {  3.0/4,   1.0/4,  5.0/4,   7.0/4   }, // p01q0
    {  7.0/12,  5.0/12, 17.0/12, 19.0/12 }, // p01q1
    { 13.0/16,  3.0/16, 19.0/16, 29.0/16 }, // p10q0
    { 15.0/16,  1.0/16, 17.0/16, 31.0/16 }, // p10q1
    { 11.0/12,  1.0/12, 13.0/12, 23.0/12 }, // p11q0
    {  3.0/4,   1.0/4,  5.0/4,   7.0/4   }, // p11q1
    };
    for (int label = 0; label < 32; label++) {
        const int row = (((label >> 3) & 1) << 2) | (((label >> 2) & 1) << 1) | (label & 1);
        const int col = (((label >> 4) & 1) << 1) | ((label >> 1) & 1);
        p[label] = polar_point(radius[RING[row]], ANGLE[row][col] * M_PI);
    }
}

void build_32apsk_4_8_4_16(points_t& p, code_rate_t rate)
{
    // Table 12e ring ratios; row = (b4, b3, b0), column = (b2, b1)
    double g1 = 2.6, g2 = 2.99, g3 = 5.6;
    switch (rate) {
    case RATE_132_180: g2 = 2.86; break;
    case RATE_7_9:
    case RATE_140_180: g1 = 2.8; g2 = 3.08; break;
    case RATE_32_45: g1 = 2.84; g2 = 2.84; g3 = 5.26; break;
    default: break;
    }
    const double r1 = std::sqrt(32.0 / (4.0 + 8.0 * g1 * g1 + 4.0 * g2 * g2 + 16.0 * g3 * g3));
    const double radius[5] = { 0.0, r1, g1 * r1, g2 * r1, g3 * r1 };
    static const int RING[8] = { 1, 4, 2, 4, 2, 4, 3, 4 };
    static const double ANGLE[8][4] = {
    {  1.0/4,   7.0/4,   3.0/4,   5.0/4   }, // 00pq0 R1
    {  7.0/16, 25.0/16,  9.0/16, 23.0/16  }, // 00pq1 R4
    {  1.0/12, 23.0/12, 11.0/12, 13.0/12  }, // 01pq0 R2
    {  1.0/16, 31.0/16, 15.0/16, 17.0/16  }, // 01pq1 R4
    {  5.0/12, 19.0/12,  7.0/12, 17.0/12  }, // 10pq0 R2
    {  5.0/16, 27.0/16, 11.0/16, 21.0/16  }, // 10pq1 R4
    {  1.0/4,   7.0/4,   3.0/4,   5.0/4   }, // 11pq0 R3
    {  3.0/16, 29.0/16, 13.0/16, 19.0/16  }, // 11pq1 R4
    };
    for (int label = 0; label < 32; label++) {
        const int row = (((label >> 4) & 1) << 2) | (((label >> 3) & 1) << 1) | (label & 1);
        const int col = (((label >> 2) & 1) << 1) | ((label >> 1) & 1);
        p[label] = polar_point(radius[RING[row]], ANGLE[row][col] * M_PI);
    }
}

void build_64apsk(points_t& p, code_rate_t rate)
{
    // Tables 13a..13f: three ring structures
    static const ring_row_t R_4_12_20_28[16] = {
    {3, {1.0/4, 7.0/4, 3.0/4, 5.0/4}},      // 0000
    {3, {13.0/28, 43.0/28, 15.0/28, 41.0/28}}, // 0001
    {3, {1.0/28, 55.0/28, 27.0/28, 29.0/28}},  // 0010
    {0, {1.0/4, 7.0/4, 3.0/4, 5.0/4}},      // 0011 (R1)
    {3, {9.0/28, 47.0/28, 19.0/28, 37.0/28}},  // 0100
    {3, {11.0/28, 45.0/28, 17.0/28, 39.0/28}}, // 0101
    {2, {1.0/20, 39.0/20, 19.0/20, 21.0/20}},  // 0110 (R3)
    {1, {1.0/12, 23.0/12, 11.0/12, 13.0/12}},  // 0111 (R2)
    {3, {5.0/28, 51.0/28, 23.0/28, 33.0/28}},  // 1000
    {2, {9.0/20, 31.0/20, 11.0/20, 29.0/20}},  // 1001 (R3)
    {3, {3.0/28, 53.0/28, 25.0/28, 31.0/28}},  // 1010
    {1, {5.0/12, 19.0/12, 7.0/12, 17.0/12}},   // 1011 (R2)
    {2, {1.0/4, 7.0/4, 3.0/4, 5.0/4}},      // 1100 (R3)
    {2, {7.0/20, 33.0/20, 13.0/20, 27.0/20}},  // 1101 (R3)
    {2, {3.0/20, 37.0/20, 17.0/20, 23.0/20}},  // 1110 (R3)
    {1, {1.0/4, 7.0/4, 3.0/4, 5.0/4}},      // 1111 (R2)
    };
    static const ring_row_t R_8_16_20_20[16] = {
    {1, {25.0/16, 23.0/16, 7.0/16, 9.0/16}},   // X0 000 R2
    {3, {7.0/4, 5.0/4, 1.0/4, 3.0/4}},      // X0 001 R4
    {1, {27.0/16, 21.0/16, 5.0/16, 11.0/16}},  // X0 010 R2
    {2, {7.0/4, 5.0/4, 1.0/4, 3.0/4}},      // X0 011 R3
    {3, {31.0/20, 29.0/20, 9.0/20, 11.0/20}},  // X0 100 R4
    {3, {33.0/20, 27.0/20, 7.0/20, 13.0/20}},  // X0 101 R4
    {2, {31.0/20, 29.0/20, 9.0/20, 11.0/20}},  // X0 110 R3
    {2, {33.0/20, 27.0/20, 7.0/20, 13.0/20}},  // X0 111 R3
    {0, {13.0/8, 11.0/8, 3.0/8, 5.0/8}},    // X1 000 R1
    {3, {37.0/20, 23.0/20, 3.0/20, 17.0/20}},  // X1 001 R4
    {1, {29.0/16, 19.0/16, 3.0/16, 13.0/16}},  // X1 010 R2
    {2, {37.0/20, 23.0/20, 3.0/20, 17.0/20}},  // X1 011 R3
    {0, {15.0/8, 9.0/8, 1.0/8, 7.0/8}},     // X1 100 R1
    {3, {39.0/20, 21.0/20, 1.0/20, 19.0/20}},  // X1 101 R4
    {1, {31.0/16, 17.0/16, 1.0/16, 15.0/16}},  // X1 110 R2
    {2, {39.0/20, 21.0/20, 1.0/20, 19.0/20}},  // X1 111 R3
    };
    static const ring_row_t R_16_16_16_16[16] = {
    {0, {1.0/16, 31.0/16, 15.0/16, 17.0/16}},  // RR00 00 R1
    {0, {3.0/16, 29.0/16, 13.0/16, 19.0/16}},  // RR00 01 R1
    {0, {7.0/16, 25.0/16, 9.0/16, 23.0/16}},   // RR00 10 R1
    {0, {5.0/16, 27.0/16, 11.0/16, 21.0/16}},  // RR00 11 R1
    {1, {1.0/16, 31.0/16, 15.0/16, 17.0/16}},  // RR01 00 R2
    {1, {3.0/16, 29.0/16, 13.0/16, 19.0/16}},  // RR01 01 R2
    {1, {7.0/16, 25.0/16, 9.0/16, 23.0/16}},   // RR01 10 R2
    {1, {5.0/16, 27.0/16, 11.0/16, 21.0/16}},  // RR01 11 R2
    {3, {1.0/16, 31.0/16, 15.0/16, 17.0/16}},  // RR10 00 R4
    {3, {3.0/16, 29.0/16, 13.0/16, 19.0/16}},  // RR10 01 R4
    {3, {7.0/16, 25.0/16, 9.0/16, 23.0/16}},   // RR10 10 R4
    {3, {5.0/16, 27.0/16, 11.0/16, 21.0/16}},  // RR10 11 R4
    {2, {1.0/16, 31.0/16, 15.0/16, 17.0/16}},  // RR11 00 R3
    {2, {3.0/16, 29.0/16, 13.0/16, 19.0/16}},  // RR11 01 R3
    {2, {7.0/16, 25.0/16, 9.0/16, 23.0/16}},   // RR11 10 R3
    {2, {5.0/16, 27.0/16, 11.0/16, 21.0/16}},  // RR11 11 R3
    };
    enum { S_4_12_20_28, S_8_16_20_20, S_16_16_16_16 } layout = S_4_12_20_28;
    double g1 = 2.4, g2 = 4.3, g3 = 7.0;
    switch (rate) {
    case RATE_7_9:
    case RATE_140_180:
    case RATE_4_5: layout = S_8_16_20_20; g1 = 2.2; g2 = 3.6; g3 = 5.2; break;
    case RATE_5_6: layout = S_8_16_20_20; g1 = 2.2; g2 = 3.5; g3 = 5.0; break;
    case RATE_128_180: layout = S_16_16_16_16; g1 = 1.88; g2 = 2.72; g3 = 3.95; break;
    default: break;
    }
    static const int COUNTS[3][4] = { { 4, 12, 20, 28 }, { 8, 16, 20, 20 }, { 16, 16, 16, 16 } };
    const ring_row_t* rows = layout == S_4_12_20_28   ? R_4_12_20_28
                             : layout == S_8_16_20_20 ? R_8_16_20_20
                                                      : R_16_16_16_16;
    const double ratio[4] = { 1.0, g1, g2, g3 };
    double energy = 0.0;
    for (int i = 0; i < 4; i++)
        energy += COUNTS[layout][i] * ratio[i] * ratio[i];
    const double scale = std::sqrt(64.0 / energy);
    float radius[4];
    for (int i = 0; i < 4; i++)
        radius[i] = (float)(ratio[i] * scale);

    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 4; col++) {
            const int pp = col >> 1, q = col & 1;
            int label;
            if (layout == S_4_12_20_28)
                label = (row << 2) | (pp << 1) | q;
            else if (layout == S_8_16_20_20)
                label = (pp << 5) | ((row >> 3) << 4) | (q << 3) | (row & 7);
            else
                label = ((row >> 2) << 4) | (q << 3) | (pp << 2) | (row & 3);
            p[label] = polar_point(radius[rows[row].ring], rows[row].angle[col] * M_PI);
        }
    }
}

void build_128apsk(points_t& p, code_rate_t rate)
{
    // Tables 14a/14b: 32 rows on six rings, label = q p b4..b0
    static const ring_row_t ROWS[32] = {
    { 0, 83.0 / 1260,     2437.0 / 1260,     1177.0 / 1260,     1343.0 / 1260 },  // 00000 R1
    { 5, 11.0 / 105,       199.0 / 105,        94.0 / 105,       116.0 / 105 },   // 00001 R6
    { 5, 37.0 / 1680,     3323.0 / 1680,     1643.0 / 1680,     1717.0 / 1680 },  // 00010 R6
    { 5, 11.0 / 168,       325.0 / 168,       157.0 / 168,       179.0 / 168 },   // 00011 R6
    { 1, 121.0 / 2520,     4919.0 / 2520,     2399.0 / 2520,     2641.0 / 2520 },  // 00100 R2
    { 2, 23.0 / 280,       537.0 / 280,       257.0 / 280,       303.0 / 280 },   // 00101 R3
    { 4, 19.0 / 720,      1421.0 / 720,       701.0 / 720,       739.0 / 720 },   // 00110 R5
    { 3, 61.0 / 720,      1379.0 / 720,       659.0 / 720,       781.0 / 720 },   // 00111 R4
    { 0, 103.0 / 560,      1017.0 / 560,       457.0 / 560,       663.0 / 560 },   // 01000 R1
    { 5, 61.0 / 420,       779.0 / 420,       359.0 / 420,       481.0 / 420 },   // 01001 R6
    { 5, 383.0 / 1680,     2977.0 / 1680,     1297.0 / 1680,     2063.0 / 1680 },  // 01010 R6
    { 5, 929.0 / 5040,     9151.0 / 5040,     4111.0 / 5040,     5969.0 / 5040 },  // 01011 R6
    { 1, 113.0 / 560,      1007.0 / 560,       447.0 / 560,       673.0 / 560 },   // 01100 R2
    { 2, 169.0 / 1008,     1847.0 / 1008,      839.0 / 1008,     1177.0 / 1008 },  // 01101 R3
    { 4, 563.0 / 2520,     4477.0 / 2520,     1957.0 / 2520,     3083.0 / 2520 },  // 01110 R5
    { 3, 139.0 / 840,      1541.0 / 840,       701.0 / 840,       979.0 / 840 },   // 01111 R4
    { 0, 243.0 / 560,       877.0 / 560,       317.0 / 560,       803.0 / 560 },   // 10000 R1
    { 5, 1993.0 / 5040,     8087.0 / 5040,     3047.0 / 5040,     7033.0 / 5040 },  // 10001 R6
    { 5, 43.0 / 90,        137.0 / 90,         47.0 / 90,        133.0 / 90 },    // 10010 R6
    { 5, 73.0 / 168,       263.0 / 168,        95.0 / 168,       241.0 / 168 },   // 10011 R6
    { 1, 1139.0 / 2520,     3901.0 / 2520,     1381.0 / 2520,     3659.0 / 2520 },  // 10100 R2
    { 2, 117.0 / 280,       443.0 / 280,       163.0 / 280,       397.0 / 280 },   // 10101 R3
    { 4, 341.0 / 720,      1099.0 / 720,       379.0 / 720,      1061.0 / 720 },   // 10110 R5
    { 3, 349.0 / 840,      1331.0 / 840,       491.0 / 840,      1189.0 / 840 },   // 10111 R4
    { 0, 177.0 / 560,       943.0 / 560,       383.0 / 560,       737.0 / 560 },   // 11000 R1
    { 5, 1789.0 / 5040,     8291.0 / 5040,     3251.0 / 5040,     6829.0 / 5040 },  // 11001 R6
    { 5, 49.0 / 180,       311.0 / 180,       131.0 / 180,       229.0 / 180 },   // 11010 R6
    { 5, 53.0 / 168,       283.0 / 168,       115.0 / 168,       221.0 / 168 },   // 11011 R6
    { 1, 167.0 / 560,       953.0 / 560,       393.0 / 560,       727.0 / 560 },   // 11100 R2
    { 2, 239.0 / 720,      1201.0 / 720,       481.0 / 720,       959.0 / 720 },   // 11101 R3
    { 4, 199.0 / 720,      1241.0 / 720,       521.0 / 720,       919.0 / 720 },   // 11110 R5
    { 3, 281.0 / 840,      1399.0 / 840,       559.0 / 840,      1121.0 / 840 },   // 11111 R4
    };
    const double ratio[6] = { 1.0, 1.715, 2.118, 2.681, 2.75, rate == RATE_135_180 ? 3.819 : 3.733 };
    const int counts[6] = { 16, 16, 16, 16, 16, 48 };
    double energy = 0;
    for (int i = 0; i < 6; i++)
        energy += counts[i] * ratio[i] * ratio[i];
    const double scale = std::sqrt(128.0 / energy);
    constexpr double PI = 3.14159265358979323846;
    for (int row = 0; row < 32; row++) {
        const float radius = (float)(ratio[ROWS[row].ring] * scale);
        for (int col = 0; col < 4; col++) {
            const int pp = col >> 1, q = col & 1;
            p[(q << 6) | (pp << 5) | row] = polar_point(radius, ROWS[row].angle[col] * PI);
        }
    }
}

void build_256apsk(points_t& p, code_rate_t rate)
{
    // Explicit coordinates (Table 15d) for 20/30 and 22/30
    const float(*coord)[2] = (rate == RATE_20_30) ? APSK256_POINTS_20_30
                             : (rate == RATE_132_180 || rate == RATE_22_30) ? APSK256_POINTS_22_30
                                                                   : nullptr;
    if (coord) {
        double energy = 0.0;
        for (int i = 0; i < 256; i++)
            energy += (double)coord[i][0] * coord[i][0] + (double)coord[i][1] * coord[i][1];
        const float scale = (float)std::sqrt(256.0 / energy);
        for (int i = 0; i < 256; i++)
            p[i] = complexf(coord[i][0] * scale, coord[i][1] * scale);
        return;
    }
    // Ring form (Tables 15a..15c): eight rings of 32 points,
    // label = XXX q p aaa, angle (2k - 1) pi / 32 mirrored by (p, q)
    double g[7] = { 1.791, 2.405, 2.980, 3.569, 4.235, 5.078, 6.536 };
    if (rate == RATE_128_180 || rate == RATE_135_180) {
        const double common[5] = { 1.794, 2.409, 2.986, 3.579, 4.045 };
        for (int i = 0; i < 5; i++)
            g[i] = common[i];
        g[5] = rate == RATE_128_180 ? 4.6 : 4.5;
        g[6] = rate == RATE_128_180 ? 5.4 : 5.2;
    }
    const double ratio[8] = { 1.0, g[0], g[1], g[2], g[3], g[4], g[5], g[6] };
    double energy = 0.0;
    for (int i = 0; i < 8; i++)
        energy += 32.0 * ratio[i] * ratio[i];
    const double scale = std::sqrt(256.0 / energy);
    static const int RING_OF_XXX[8] = { 0, 1, 3, 2, 7, 6, 4, 5 };
    static const int K_OF_AAA[8] = { 1, 2, 4, 3, 8, 7, 5, 6 };
    const double pi = M_PI;
    for (int xxx = 0; xxx < 8; xxx++) {
        const float radius = (float)(ratio[RING_OF_XXX[xxx]] * scale);
        for (int q = 0; q < 2; q++)
            for (int pp = 0; pp < 2; pp++)
                for (int aaa = 0; aaa < 8; aaa++) {
                    const double phi = (2 * K_OF_AAA[aaa] - 1) * pi / 32.0;
                    const double angle = pp ? (q ? pi + phi : pi - phi) : (q ? -phi : phi);
                    p[(xxx << 5) | (q << 4) | (pp << 3) | aaa] = polar_point(radius, angle);
                }
    }
}

bool is_16apsk_8_8(code_rate_t rate)
{
    return rate == RATE_90_180 || rate == RATE_96_180 || rate == RATE_100_180 || rate == RATE_18_30 ||
           rate == RATE_20_30;
}

bool is_32apsk_rb(code_rate_t rate) { return rate == RATE_2_3 || rate == RATE_32_45; }

bool is_32apsk_4_8_4_16(code_rate_t rate)
{
    return rate == RATE_128_180 || rate == RATE_132_180 || rate == RATE_22_30 || rate == RATE_140_180 ||
           rate == RATE_7_9;
}

} // namespace

constellation_t make_constellation(modulation_t modulation,
                                   code_rate_t rate,
                                   fecframe_size_t framesize)
{
    constellation_t c;
    switch (modulation) {
    case MOD_QPSK: c.bits_per_sym = 2; build_qpsk(c.points); break;
    case MOD_8PSK: c.bits_per_sym = 3; build_8psk(c.points); break;
    case MOD_8APSK: c.bits_per_sym = 3; build_8apsk(c.points, rate); break;
    case MOD_16APSK:
        c.bits_per_sym = 4;
        if (is_16apsk_8_8(rate))
            build_16apsk_8_8(c.points, rate);
        else
            build_16apsk_4_12(c.points, rate);
        break;
    case MOD_32APSK:
        c.bits_per_sym = 5;
        if (is_32apsk_rb(rate))
            build_32apsk_rb(c.points, rate, framesize);
        else if (is_32apsk_4_8_4_16(rate))
            build_32apsk_4_8_4_16(c.points, rate);
        else
            build_32apsk_4_12_16(c.points, rate);
        break;
    case MOD_64APSK: c.bits_per_sym = 6; build_64apsk(c.points, rate); break;
    case MOD_128APSK: c.bits_per_sym = 7; build_128apsk(c.points, rate); break;
    case MOD_256APSK: c.bits_per_sym = 8; build_256apsk(c.points, rate); break;
    default: throw std::invalid_argument("make_constellation: unsupported modulation");
    }
    c.n_points = 1 << c.bits_per_sym;
    return c;
}

int nearest_label(const constellation_t& c, complexf sym)
{
    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int label = 0; label < c.n_points; label++) {
        const float d = std::norm(sym - c.points[label]);
        if (d < best_dist) {
            best_dist = d;
            best = label;
        }
    }
    return best;
}

} // namespace cusat_dsp

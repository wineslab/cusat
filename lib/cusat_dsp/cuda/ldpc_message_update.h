/*
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_MESSAGE_UPDATE_H
#define INCLUDED_CUSAT_DSP_LDPC_MESSAGE_UPDATE_H

#include <cstdint>

#ifdef __CUDACC__
#define CUSAT_LDPC_HD __host__ __device__
#define CUSAT_LDPC_INLINE __forceinline__
#else
#define CUSAT_LDPC_HD
#define CUSAT_LDPC_INLINE inline
#endif

namespace cusat_dsp {
namespace cuda {

struct ldpc_bnl_update_t {
    int8_t message;
    int delta;
};

CUSAT_LDPC_HD CUSAT_LDPC_INLINE int8_t ldpc_clamp_bnl_message(int v)
{
    return (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
}

CUSAT_LDPC_HD CUSAT_LDPC_INLINE ldpc_bnl_update_t
ldpc_make_bnl_update(int raw_new_message, int old_message)
{
    const int8_t msg = ldpc_clamp_bnl_message(raw_new_message);
    return { msg, (int)msg - old_message };
}

} // namespace cuda
} // namespace cusat_dsp

#undef CUSAT_LDPC_HD
#undef CUSAT_LDPC_INLINE

#endif /* INCLUDED_CUSAT_DSP_LDPC_MESSAGE_UPDATE_H */

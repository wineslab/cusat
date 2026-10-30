/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Runtime loader for the radio driver plugin (libradio_uhd.so). Keeps the
 * GPLv3 USRP driver out of this Apache-2.0 application.
 */

#pragma once

#include "trx/radio_device.h"

#include <memory>

/**
 * dlopen the radio plugin "libradio_<driver>.so" (driver: "uhd" or "difi";
 * overridable via the CUSAT_RADIO_PLUGIN env var), resolve
 * radio_device_create() and instantiate a device. Returns nullptr on
 * failure (logs to stderr). The plugin stays mapped for process lifetime.
 */
std::unique_ptr<radio_device> load_radio_device(const std::string& driver = "uhd");

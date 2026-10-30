# Functional split dashboard preview

Open http://localhost:8766 when the `cusat-split-preview` tmux session is running. To serve manually:

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory dashboard-preview
```

Standalone interface: no API calls, telemetry or hardware control. TX blocks point right and RX blocks left. Drag the larger split handles continuously; on release they snap to the nearest supported boundary. Placement commits on release, and cancelled drags restore the previous selection. Select marked positions, or use arrows/Home/End when focused. Underbraces identify Spark and X410 ownership. RX I/Q precision (16 or 8 bits) is grouped with processing placement. Restore baseline is in the header.

## RX timing split

RX has three boundaries: sample IQ (4 samples/symbol), filtered IQ (planned 2 samples/symbol), and synchronized symbols (1 sample/symbol). The synchronized-symbol boundary moves matched filtering and timing recovery to the FPGA; carrier recovery and frame/pilot recovery stay on the host. Click the Timing recovery arrow itself to toggle Gardner or feedforward energy-search timing without changing placement or the payload budget. The algorithm label and block accent change visibly; keyboard Enter/Space also work. The button is inactive when timing runs on the host. Feedforward is the default; Gardner requires input above approximately 2.1 samples/symbol, whereas feedforward supports 1.2–7. Both are source-level candidate capabilities, not end-to-end radio validation. This disconnected UI switches immediately. Hardware switching requires stopping/draining streams, loading algorithm-specific coefficients/registers, adapting the host receive path and reacquiring, without a bitstream reload if the candidate image validates.

## Satellite controls

Two permanent columns sit outside the animation box, to its right. Their headings, readouts, track limits and toggles share aligned positions.

- **Altitude:** always adjustable, independently of animation, from 550 km to GEO at 35,786 km. A logarithmic slider gives low-orbit heights useful resolution; the visible readout and accessible value report kilometers. The switch below starts/pauses animation without changing altitude.
- **MODCOD:** ACM is the default and shows the selected MODCOD as a vertical thermometer with adjustable minimum/maximum limits. CCM transforms the same column into a vertical slider and holds that choice fixed, retaining the green meter and a green square handle. Limits cannot cross.
- **Baseline:** restores the animated 550-km orbit, ACM with full bounds, and 16-bit RX precision.

There is no elevation, SNR or pass-position slider. SNR derives from geometry. Changes to altitude and ACM/CCM can be combined independently with animation running or paused.

## Metrics and models

**DIFI TX Tput / DIFI RX Tput:** calculated payload budgets, excluding framing/pilots and DIFI/network overhead. Assumptions: 15.36 Msym/s, sample IQ at 4 samples/symbol, planned filtered RX at 2 samples/symbol. The latter is a proposed configuration, not validation of the current deployment.

**Pilot SNR:** a separately labeled modeled SNR graph, not a pilot measurement. Earth radius is 6371 km; altitude is the selected height. SNR is 22 dB at 550-km zenith minus `20 log10(slant distance / 550 km)`, representing relative free-space loss with fixed assumed link budget. Doppler, fading, atmosphere, antenna patterns and RF calibration are excluded. High altitudes can put this model below the guarded ACM minimum; this does not predict a calibrated GEO system's performance.

**OTA throughput:** a graph of the calculated coded-rate ceiling (symbol rate × constellation bits × code rate), with current MODCOD and ACM/CCM indication. It drops to zero outside line of sight. It excludes framing/pilot overhead and decoding losses, so it is not measured goodput. There are eight selectable MODCODs. ACM uses their measured thresholds from `lib/cusat_dsp/acm_config.h` plus the default 2.5-dB guard. This UI does not reproduce controller hysteresis, OLLA, probation or feedback. Below the selected minimum's threshold it retains that floor and displays a warning; it does not predict FER or usable goodput.

**Ping:** calculated propagation-only RTT for a ground–satellite–ground link, `4 × slant distance / c`. Queueing, processing and protocol delays are excluded. Zenith examples: approximately 7.3 ms at 550 km and 477.5 ms at GEO. This gauge is not measured ICMP ping. When line of sight is lost it is unavailable.

**FER:** a gauge beside Ping reserved for post-FEC frame error telemetry. It shows “Not connected” and an unavailable value; this preview has no validated FER model.

A complete animated orbit takes 10 seconds at any selected altitude. Its no-LoS part is 24× faster than visible motion, with piecewise rates normalized to the total duration. Link loss hides the ground link, displays **Link lost**, leaves SNR graph gaps and greys the ACM thermometer. The selected MODCOD is retained during an outage. On reacquisition, normal speed and automatic selection resume. DIFI gauges continue to describe the chosen fronthaul payload format.

## Visual scale and provenance

Earth stays fixed in size. Drawn orbital radius varies gently and logarithmically over the altitude range, preventing GEO from making Earth tiny or sending the satellite outside the square. Physics calculations use actual altitude and slant distance; only the schematic drawing is compressed. Globe and satellite are code-native SVG drawings with muted slate/blue styling.

Orbit inspiration comes from `/Users/maxime/Documents/cuSat---Wintech/dvb-s2x/deploy/dashboard/`. Its archived route is `aerial/demo-dashboard`, at `https://demo-dashboard-aerial.apps.tenoran.automation.otic.open6g.net`; on inspection it had zero replicas and returned 503. No OpenShift resources were changed.

The mapping/pilot/scrambling stages move together because the candidate image uses one `dvb_plmod` block. Its input carries literal PLHEADER symbols and packed indices; its output has mapped, pilot-inserted, scrambled symbols. No intermediate selectable symbol entry exists within that block in this candidate image.

The preview uses system fonts. Processing stages follow Figure 3 of the cuSAT paper.

## Validation

The dark panel layout fills desktop viewports; smaller screens stack and scroll. Browser checks cover 1920×1080, 1366×768, 1470×956 and mobile width at LEO and GEO. Checks include altitude endpoints, modeled zenith RTT, pause/resume independent of altitude, min/max ACM bounds, manual CCM hold, outages, reset, compression/splits, graph rendering and absence of API calls. A real-time 10-second orbit check verifies multiple MODCOD states, both LoS states and return to the initial phase. ACM/CCM track alignment is checked. Reduced-motion preferences disable easing transitions.

This is a standalone Grafana-inspired UI, not a deployed Grafana instance or hardware-connected dashboard.

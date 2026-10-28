# MaintIQ glossary

## Alert levels
- Critical: latest window is Isolation Forest flagged, or classifier predicted failure-within-24h, or calibrated probability ≥ 0.315.
- High: Isolation Forest anomaly flag on a scored window.
- Medium / Low: lesser stored flags. Statuses: New, Acknowledged, Investigating, Resolved.

## Health
- Critical / Watch / Healthy / Unknown come from the latest scored window, not from live certification.

## Features
- RMS, peak, kurtosis: time-domain vibration on 12-hour bins.
- Band energy: FFT bands 0–500, 500–1000, 1000–2000, 2000–4000 Hz.
- Fault frequencies at 2000 RPM: FTF, BSF, BPFO, BPFI.

## IDs
- Public machine id examples: brg-003, t2-b1.
- Asset IDs: BRG-003, T2-B1.

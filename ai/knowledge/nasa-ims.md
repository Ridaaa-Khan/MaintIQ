# NASA IMS dataset as used in MaintIQ

MaintIQ treats the scored NASA IMS bearings as an uploaded monitoring fleet.

## Structure
- Three run-to-failure tests (1st_test, 2nd_test, 3rd_test), four bearings each — 12 trajectories.
- Each snapshot: 20,480 samples, 20 kHz.
- Shaft speed used for fault frequencies: 2000 RPM.
- UI aggregates non-artifact rows into 12-hour bins (41 / 15 / 91 bins for tests 1 / 2 / 3).
- Operational snapshot counts stored with the runs: 2155 (1st), 981 (2nd), 6322 (3rd).

## Documented damage (labels, not health badges)
- T1-B3: inner race
- T1-B4: rolling element
- T2-B1: outer race
- T3-B3: outer race (UI id brg-003, asset BRG-003)
- T3-B4: no documented failure (UI id brg-004, asset BRG-004)

## UI identifiers
- Keys: t1-b1 … t3-b2, brg-003, brg-004.
- Assets: T1-B1 … T3-B2, BRG-003, BRG-004.
- Names: "IMS 1st_test Bearing N" except the two Test 3 held-out bearings, labeled "IMS Bearing 3" and "IMS Bearing 4".

## What is not in the dataset
- No live plant, no weather, no operator notes beyond what a user typed.
- No invented sensor channels. If a value is not in the scored files or the machines table, it is unavailable.

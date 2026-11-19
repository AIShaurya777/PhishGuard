# External data (Milestones 7 and 7b)

## Popular legitimate websites (Milestone 7)

Used to **test** saved models on popular real websites outside PhiUSIIL. In Milestone 7 nothing here was used for training; Milestone 7b's decision entry says exactly which parts it trains on and which it keeps for testing.

| File | Source | Content | Downloaded | SHA-256 of this file |
|---|---|---|---|---|
| `crux_top10k_2026-10.csv` | Chrome UX Report (CrUX) global top origins, via [zakird/crux-top-lists](https://github.com/zakird/crux-top-lists) `data/global/current.csv.gz` (last updated 2026-10-13) | the 10,000 most-visited **origins** in Chrome (rank buckets 1k / 5k / 10k), e.g. `https://accounts.google.com`, **including subdomains** | 2026-10-27 | `ffd7eb94e50948249671c54b5f1165aef5a7f8ba20e24cd8cccd5a9fa334c659` |
| `tranco_GQ3QK_top20k.csv` | [Tranco](https://tranco-list.eu/list/GQ3QK) list `GQ3QK` (2026-09-27 → 2026-10-26, Dowdall combination of CrUX, Farsight, Majestic, Radar, Umbrella; registered domains only) | the 20,000 top-ranked **registered domains** (a header row was added) | 2026-10-27 | `cb8fa27203729f23bfb53fde4d19771092fcc1253794fb7b7cdfe0d05e147441` |

Source files before extraction: CrUX `current.csv.gz` SHA-256 `be444a34…53011f88`; Tranco download SHA-256 `7252df44…1d32f64f`.

**Caveat:** "popular" isn't the same as "safe". A few top sites may be gambling, adult or file-sharing sites, but they aren't phishing pages, so we count them as legitimate.

**Citations:**
- V. Le Pochat et al., "Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation", *NDSS* 2019.
- Google, Chrome UX Report.

## Phishing feeds (Milestone 7b), snapshot of 2026-11-19

Fresh phishing from outside PhiUSIIL, frozen on one day so results can be reproduced (the live feeds change daily). **These are real malicious links: never open them in a browser.**

| File | Source | Content | SHA-256 of this file |
|---|---|---|---|
| `phishtank_online-valid_2026-11-19.csv.gz` | [PhishTank](https://phishtank.org) `online-valid.csv` | 43,703 community-verified phishing URLs still online, with submission dates (mostly 2025–2026) and target brand | `3ed127e58173ff6c782253159c2c042745c356ceba85c84c3997e71bd156e003` |
| `phishing-database_domains-active_2026-11-19.txt.gz` | [Phishing.Database](https://github.com/Phishing-Database/Phishing.Database) `phishing-domains-ACTIVE.txt` | 392,178 phishing domains confirmed active, no dates | `65366d923df4f53619637da02f9fe74825c8e9190ebdcaca9b7de0b3893ff53e` |
| `openphish_feed_2026-11-19.txt` | [OpenPhish](https://openphish.com) community feed | 300 most recent phishing URLs | `17c99002a2099b3320fb9810b7465e26db6696428bbe01e8d5c47c1642adc197` |

Downloaded 2026-11-19; compressed with `gzip -n -9`. PhishTank and OpenPhish data are provided for research use; cite the sources.

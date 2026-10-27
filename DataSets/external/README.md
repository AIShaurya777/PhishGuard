# External lists of legitimate websites (Milestone 7)

Used only to **test** saved models on popular real websites outside PhiUSIIL. Nothing here is used for training.

| File | Source | Content | Downloaded | SHA-256 of this file |
|---|---|---|---|---|
| `crux_top10k_2026-10.csv` | Chrome UX Report (CrUX) global top origins, via [zakird/crux-top-lists](https://github.com/zakird/crux-top-lists) `data/global/current.csv.gz` (last updated 2026-10-13) | the 10,000 most-visited **origins** in Chrome (rank buckets 1k / 5k / 10k), e.g. `https://accounts.google.com`, **including subdomains** | 2026-10-27 | `ffd7eb94e50948249671c54b5f1165aef5a7f8ba20e24cd8cccd5a9fa334c659` |
| `tranco_GQ3QK_top20k.csv` | [Tranco](https://tranco-list.eu/list/GQ3QK) list `GQ3QK` (2026-09-27 → 2026-10-26, Dowdall combination of CrUX, Farsight, Majestic, Radar, Umbrella; registered domains only) | the 20,000 top-ranked **registered domains** (a header row was added) | 2026-10-27 | `cb8fa27203729f23bfb53fde4d19771092fcc1253794fb7b7cdfe0d05e147441` |

Source files before extraction: CrUX `current.csv.gz` SHA-256 `be444a34…53011f88`; Tranco download SHA-256 `7252df44…1d32f64f`.

**Caveat:** "popular" isn't the same as "safe". A few top sites may be gambling, adult or file-sharing sites, but they aren't phishing pages, so we count them as legitimate.

**Citations:**
- V. Le Pochat et al., "Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation", *NDSS* 2019.
- Google, Chrome UX Report.

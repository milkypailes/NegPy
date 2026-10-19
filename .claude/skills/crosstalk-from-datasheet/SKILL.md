---
name: crosstalk-from-datasheet
description: "Use when the user supplies a film-stock spec sheet, datasheet, or spectral data (PDF / URL / numbers) and wants a NegPy spectral-crosstalk .toml matrix for the Process panel's Crosstalk control, or asks to calibrate / derive channel-unmixing for a specific film or scanner. Keywords: crosstalk, datasheet, spec sheet, film stock, dye density, spectral, separation, unmixing, toml."
---

# Crosstalk from datasheet

## Overview

A color negative's three dye layers each leak density into channels they shouldn't
("unwanted absorption"), muddying color. NegPy's **Crosstalk** control (Process panel)
unmixes them with a 3×3 matrix applied to the raw negative densities. This skill turns a
film's published **spectral dye-density** curves into that matrix: read the unwanted
absorptions, invert them, write a `.toml`.

**Core principle:** the matrix is the *inverse* of the dye-leakage matrix. You only read
values off curves and judge data quality — `derive_matrix.py` does the arithmetic.

Read `docs/CROSSTALK.md` and `crosstalk/README.md` once for the file format and the
density-domain math NegPy applies (`d=-log10(rgb_negative)`, `d_out=M·d`, pre-normalization);
NegPy **row-normalizes** `M` and blends with identity by the Crosstalk strength (0–1), so
only the off-diagonal/diagonal *ratios* matter — absolute row scale is irrelevant.

## Workflow

1. **Acquire the source.**
   - Local PDF → `Read` it with `pages` (**vision** — the curves are pixels; a text fetch
     will not see them).
   - URL → PDF → `curl` it to the scratchpad, then `Read` the pages as images.
   - HTML page → `WebFetch` for any numeric spectral tables; if it only embeds a plot,
     download the image and `Read` it.
   - Raw spectral numbers / CSV → use them directly (Tier 1).

2. **Classify the data tier** (the one decision that's easy to get wrong):

   ```dot
   digraph tier {
     "Separated C/M/Y dye-density curves\n(or numeric spectral data)?" [shape=diamond];
     "Only aggregate neutral dye-density\n(one combined curve)?" [shape=diamond];
     "Tier 1: derive directly" [shape=box];
     "Tier 2: estimate, label (approx)" [shape=box];
     "Tier 3: do NOT fabricate" [shape=box];
     "Separated C/M/Y dye-density curves\n(or numeric spectral data)?" -> "Tier 1: derive directly" [label="yes"];
     "Separated C/M/Y dye-density curves\n(or numeric spectral data)?" -> "Only aggregate neutral dye-density\n(one combined curve)?" [label="no"];
     "Only aggregate neutral dye-density\n(one combined curve)?" -> "Tier 2: estimate, label (approx)" [label="yes"];
     "Only aggregate neutral dye-density\n(one combined curve)?" -> "Tier 3: do NOT fabricate" [label="no"];
   }
   ```

   Most *consumer* datasheets (e.g. Kodak Gold 200) are **Tier 2**: the spectral-dye-density
   plot is one *aggregate* "midscale neutral" curve, not separated dyes. Spectral-*sensitivity*
   curves are **not** dye-density and must never be used as a substitute.

3. **Read the leakage values into matrix `A`** (rows = R/G/B measurement bands, cols = C/M/Y
   dyes). Dye→channel: **cyan↔Red, magenta↔Green, yellow↔Blue**. Sample each dye's diffuse
   density at band centers **R≈650 / G≈550 / B≈450 nm** (datasheets use Status M). Each dye's
   own-band value is the diagonal; the other two are its unwanted absorption.
   Reading a graph by eye is ±~0.05 density — fine, the result gets row-normalized.

   Read the dyes **above D-min**. Kodak motion-picture sheets say "D-mins subtracted" on the
   plot; if a sheet draws each dye over the Minimum Density curve, subtract that curve first.
   A net curve includes the masking couplers, so it can dip slightly **below zero** in another
   band (Vision3 magenta at 450 nm): read that value as negative, do not clamp it to 0.

   - **Tier 1:** read all nine values off the separated curves.
   - **Tier 2:** trace the midscale-neutral and Minimum Density curves into
     `scripts/crosstalk_dyes/<film>_neutral.csv` (`wavelength_nm,midscale,minimum`) and run
     `uv run python scripts/fit_crosstalk_estimates.py`. It fits their difference with Kodak's
     Vision3 dye curves and prints the readings. Do not guess the off-diagonals from dye
     chemistry: the masking couplers of a C-41 or ECN-2 negative cancel most of the unwanted
     absorptions that the dyes alone have. The estimate assumes the film's masking matches
     Vision3's, which fits Kodak films best. Name it `"… (approx)"`.
   - **Tier 3:** stop. State that the datasheet lacks dye-density data. Optionally emit a
     conservative starter close to NegPy's Default, named `"… (approx, starter)"`, and suggest
     chart-based calibration instead. Never invent precision.

4. **Run the deriver:**
   ```bash
   echo '{"name":"Kodak Gold 200 (approx)","readings":[[1.0,0.06,0.02],[0.04,1.0,0.05],[0.02,0.12,1.0]]}' \
     | python3 .claude/skills/crosstalk-from-datasheet/derive_matrix.py --out <film>.toml
   ```
   `readings` is `A` row-major (R/G/B band rows). It column-normalizes, inverts, and rescales
   so the diagonal reads `1.0`; it **exits non-zero** on a singular or misshaped input rather
   than emitting a bad matrix.

5. **Review & self-validate** — never trust the loader's silence (it skips malformed files
   without error):
   - 3×3 numeric, no booleans.
   - Diagonal `1.0`; off-diagonals mostly **negative** (a correction *subtracts* contamination)
     and small — flag anything `|·| > 0.2` as a likely misread/transpose. A small positive term
     (up to ~0.05) is correct where a reading was negative: the masking coupler removed more
     density in that band than the dye added.
   - Confirm it actually loads:
     ```bash
     uv run python -c "from negpy.services.assets.crosstalk import CrosstalkProfiles as C; \
       r=C._parse_file('<film>.toml'); assert r, 'INVALID — would be skipped'; print('ok', r[0])"
     ```

6. **Write the file** — pick the destination by intent:
   - **Contribute / bundle** → repo gallery `crosstalk/<film>.toml`.
   - **Personal use now** → `<Documents>/NegPy/crosstalk/<film>.toml` (resolve the Documents
     path via `get_default_user_dir()` in `negpy/kernel/system/paths.py`).

## Quick reference

| | |
|---|---|
| TOML keys | `matrix` (required, 3×3 numeric, no bool); `name` (optional; `"Default"` reserved) |
| Convention | rows = **output** channels, cols = **input** channels (R/G/B) |
| Band centers | R 650 nm · G 550 nm · B 450 nm (Status M) |
| Dye→channel | cyan→R · magenta→G · yellow→B |
| `A` shape | rows = bands (R/G/B), cols = dyes (C/M/Y); diagonal = own-band peak |
| Default (fallback) | `[1.0,-0.05,-0.02, -0.04,1.0,-0.08, -0.01,-0.1,1.0]` |

## Common mistakes

- **Transposing rows/cols** — `A` is bands×dyes; the output is out-channels×in-channels. Keep
  R/G/B order throughout.
- **Large positive off-diagonals in the output** — wrong sign; a correction must *subtract*
  leakage. Only a below-zero reading on a net curve gives a small positive term.
- **Reading over D-min** — a dye curve drawn on top of the Minimum Density curve inflates every
  unwanted absorption; subtract D-min first.
- **Using sensitivity curves as dye density** — they measure different things; not interchangeable.
- **Over-claiming precision** — an aggregate (Tier 2) curve gives an estimate; say so in the name.
- **Forgetting it's density-domain** — read *density* off the dye-density plot, not transmittance.

## Worked example — Kodak Gold 200 (Tier 2)

`Read` the datasheet PDF pages as images. The CURVES section has Characteristic, Spectral-
Sensitivity, and Spectral-Dye-Density plots — but the dye-density plot holds only the
*aggregate* midscale neutral and the Minimum Density curve → **Tier 2**. Kodak's PDFs draw
both as vector paths: read them exactly into `scripts/crosstalk_dyes/kodak_gold_200_neutral.csv`
and run the estimator, which prints:

```
{"name": "kodak_gold_200", "readings": [[0.7727, 0.069, 0.0056], [0.0797, 0.9412, 0.0686], [0.0804, -0.0046, 1.0018]]}
```

Pipe that line to the deriver → emitted TOML (note `name` flags it approximate):

```toml
name = "Kodak Gold 200 (approx)"
matrix = [
  [1.0000,  -0.0733,  -0.0006],
  [-0.0961,   1.0000,  -0.0679],
  [-0.1054,   0.0126,   1.0000],
]
```

The magenta → blue term is small and positive: Vision3's masking couplers remove more blue
density than the magenta dye adds. Validate it loads, then write it to
`crosstalk/kodak_gold_200.toml` (contribute) or the Documents folder (personal).

# Crypto Derivatives Microstructure Research Plan v1.0

**Mission:** Discover, replicate, and critically evaluate statistically significant relationships between funding rates, open interest, liquidations, and short-horizon price moves in BTC and ETH perpetual futures across major exchanges (~2019–present).

**Target User:** Quantitative trading firm seeking deep understanding of crypto microstructure signals, niche data sources, validated reproducible results, and reusable data-processing tools.

---

## Core Research Questions

1. **Funding Rate Predictability:** Do funding rate levels or changes predict future returns on BTC/ETH perps or spot?
2. **Open Interest Dynamics:** How do open interest and long/short positioning imbalances interact with funding to predict returns?
3. **Liquidation Events:** Are extreme liquidation events predictive of short-term reversals or momentum?
4. **Exchange-Specific Effects:** Which microstructure relationships are exchange-dependent vs. universal?
5. **Transaction Cost Robustness:** Do identified effects remain economically meaningful after realistic transaction costs?

---

## High-Level Hypotheses (Testable Claims)

- **H1:** Positive funding rates predict negative next-period perp returns (mean reversion of funding excesses).
- **H2:** Sharp increases in open interest, combined with negative funding, predict liquidations and subsequent reversals.
- **H3:** Long/short imbalance extremes (z-score > 2) predict 1-4h directional moves in the opposite direction.
- **H4:** Exchange-specific idiosyncrasies exist (e.g., Binance vs. Bybit) in funding/open-interest relationships.
- **H5:** Spot-perp basis anomalies + high open interest changes predict intraday volatility spikes.

---

## Data Priorities

### Primary Datasets
1. **Funding Rates:** Historical funding rates, accrual times, exchange-specific rates (Binance, Bybit, OKX, Deribit).
2. **Open Interest & Positioning:** Aggregate long/short ratios, OI trends, exchanges supporting these metrics.
3. **Liquidations:** Aggregated or per-event liquidation data (buy/sell volumes, price), major exchanges.
4. **OHLCV Data:** Spot and perpetual bars (1m–1d granularity), BTC/ETH, all major exchanges.
5. **Exchange Metadata:** Trading hours, fee schedules, settlement cycles, known outages.

### Secondary Datasets (if time permits)
- On-chain metrics (whale movements, exchange flows, long/short positioning from futures data).
- Options implied volatility (if available).
- Cross-exchange arbitrage opportunities.

---

## Proposed Phases & Deliverables

### Phase 0: Scoping & Planning (Current)
- **Deliverable:** research_plan_v1.md (this file)
- **Status:** In Progress

### Phase 1: Data Discovery
- Identify ≥3 public/semi-public sources for funding, OI, liquidations, OHLCV.
- Document schema, rate limits, time coverage, known caveats.
- **Deliverable:** docs/data_sources.md

### Phase 2: Data Acquisition
- Implement Python downloaders for each source.
- Handle pagination, retries, rate limits.
- Store raw data in `/data/raw/`.
- **Deliverable:** Python scripts in `/src/crypto_perp_research/`, raw CSV/Parquet files.

### Phase 3: Data Engineering & Integration
- Normalize symbols, timestamps, time zones (UTC).
- Build canonical research tables (funding_panel, oi_panel, returns_panel, liquidations_panel).
- Write tests for data quality.
- **Deliverable:** `/src/crypto_perp_research/` with loaders, cleaners, feature engineers; `/tests/test_*.py`

### Phase 4: Exploratory Data Analysis
- Visualize distributions, regimes, rolling correlations.
- Identify structural breaks, missing data, anomalies.
- **Deliverable:** EDA notebooks/scripts in `/analysis/`, figures in `/analysis/figures/`

### Phase 5: Hypothesis Testing & Modeling
- Fit baseline models (OLS, logistic regression) for each hypothesis.
- Evaluate train/test by time, report coefficients, t-stats, p-values.
- Explore nonlinear models (trees, simple ensemble) if linear signals are weak.
- **Deliverable:** Model results in `/analysis/results/`, intermediate tables in `/data/processed/`

### Phase 6: Robustness & Validation
- Challenge assumptions: different periods, exchanges, vol regimes.
- Apply multiple-hypothesis corrections (Bonferroni / BH-FDR).
- Compute simple transaction-cost adjustments.
- **Deliverable:** docs/validation_notes.md, "red team" critique of findings.

### Phase 7: Reporting & Packaging
- Assemble comprehensive report: executive summary, methodology, key findings, limitations, reproducibility.
- Ensure all figures/tables reproducible from code.
- **Deliverable:** reports/crypto_perp_funding_research.md, updated README.md

---

## Key Milestones

| Phase | Milestone | Target Completion | Owner |
|-------|-----------|-------------------|-------|
| 0 | Research plan drafted | Day 1 | Research Director |
| 1 | ≥3 data sources documented | Day 1–2 | Data Discovery |
| 2 | Raw data downloaded & stored | Day 2–3 | Data Acquisition |
| 3 | Cleaned canonical tables & tests passing | Day 3–4 | Data Engineering |
| 4 | EDA plots & insights | Day 4–5 | Quant Researcher |
| 5 | Model results & statistical tests | Day 5–7 | Quant Researcher |
| 6 | Robustness report & signal filtering | Day 7–8 | Validation |
| 7 | Final report & reproducibility verified | Day 8–9 | Reporting |

---

## Success Criteria

- [x] Multi-year historical data for BTC and ETH funding, OI, liquidations.
- [ ] At least 1 statistically significant, economically meaningful signal that survives robustness checks.
- [ ] Clean, tested, reusable Python modules for data loading and feature engineering.
- [ ] Comprehensive documentation and reproducibility instructions.
- [ ] Clear, honest assessment of limitations and caveats.

---

## Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Data fragmentation across exchanges | Normalize schema, test for consistency, document exchange-specific quirks. |
| Rate limits & access restrictions | Prioritize free APIs and bulk downloads; cache locally. |
| Time-zone and timestamp alignment | Convert all to UTC; test for gaps and duplicates. |
| Overfitting & data mining bias | Use time-series splits; apply multiple-hypothesis corrections. |
| Transaction costs reducing edge | Model slippage, fees, and funding payments explicitly. |

---

## Tools & Environment

- **Languages:** Python 3.x
- **Key Libraries:** pandas, numpy, seaborn, matplotlib, statsmodels, scikit-learn, requests
- **Testing:** pytest
- **Version Control:** git
- **Project Layout:** /data, /src, /analysis, /tests, /docs, /reports

---

## Next Steps

1. **Immediate (Phase 1):** Launch Data Discovery Agent to search for and document data sources.
2. **Then (Phase 2):** Implement data acquisition scripts.
3. **Iterate:** Follow the phased approach, maintaining event log and periodic checkpoints.

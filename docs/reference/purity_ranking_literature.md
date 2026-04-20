# Purity Ranking: Literature Context

**Date:** 2026-04-17
**Context:** Incident-validation analysis — purity-ranked vs stability-ranked saturation comparison

## The Method

For each protein, coefficient drift is computed between two training conditions (incident-only vs. incident+prevalent):

```
noise_score(p)  = |μ_IP − μ_IO| / (|μ_IO| + ε)
purity_score(p) = |μ_IO(p)| / (1 + noise_score(p))
```

`combined_purity_bw` = mean of min-max normalized L1 and L2 purity scores. Used to re-rank proteins for panel build-up (saturation analysis), prioritizing proteins whose signal is specific to pre-diagnostic risk.

## Verdict

**No direct precedent.** The specific operationalization — per-feature coefficient drift ratio across case-definition training conditions, used to re-rank a saturation panel — is novel. The concern it addresses is well-established; the feature-level operationalization is not.

## Literature Map

### Motivating Framing (cite for the problem)

- **Ransohoff 2004** (Nat Rev Cancer, PMID 15685197) — canonical paper establishing that prevalent case contamination hard-wires bias into downstream biomarker comparisons. PRoBE framework requires prospective pre-diagnosis collection. Fix is sample exclusion; our fix is feature re-weighting.
- **Ransohoff 2010** (JCO, PMID 20038718) — extends to specimen-level biases; statistical analysis cannot correct baseline inequality post-hoc.

### Empirical Near-Precedents (cite as closest empirical analogs)

- **Kachuri et al. 2024** (Nat Commun) — stratifies 618 protein–cancer associations into <3yr, 3–7yr, >7yr to diagnosis; proteins persisting at >7yr treated as pre-diagnostic risk factors vs. reverse causation candidates. Time-strata persistence is the closest published proxy to our purity score, but no formal drift coefficient is computed.
- **Robbins et al. 2023** (Nat Commun, PMID 37264016) — 392–1,162 proteins in blood up to 3yr pre-diagnosis across 6 prospective cohorts; identifies 19 proteins with heterogeneous association strength by lead time (EN-RAGE: OR 1.10 at 2–3yr vs. 2.49 at <1yr). These high-drift proteins are functionally equivalent to high noise_score proteins in our framework.
- **Johansson et al. 2023** (eBioMedicine, PMID 37379654) — finds only 30% overlap in protein sets selected at 1–3yr vs. 1–5yr pre-diagnosis; interprets temporally stable proteins as markers of inherent risk. Same scientific motivation, observational stratification rather than formal scoring.

### Methodological Near-Precedents (cite for the scoring approach)

- **StableMate 2024** (NAR Genomics, PMID 39345755) — selects features whose predictive association is consistent across heterogeneous environments (e.g., disease status, study cohort). Closest published method: environment-conditioned feature stability used to rank and filter features. Key difference: uses selection frequency consistency, not coefficient magnitude drift.
- **Stabl 2024** (Nat Biotechnol, PMID 38168992) — features ranked by selection frequency on real vs. noise-injected data. Single training condition; stability-based ranking only.

### Theoretical Grounding

- **Peters et al. 2016** (JRSS-B, arXiv 1501.01332, Invariant Causal Prediction) — causal features are those whose regression coefficient on the response is invariant across experimental environments. Features that drift across environments are flagged as non-causal. Our purity score is a one-sided ICP-inspired stability measure: proteins whose coefficient drifts between incident-only and incident+prevalent training are, by ICP logic, not capturing pre-diagnostic causal risk.

## Framing for the Paper

Present purity ranking as a **novel feature-level operationalization** of a well-established epidemiological concern (Ransohoff 2004/2010), grounded theoretically in invariant prediction (Peters 2016), with the closest empirical analogs being Kachuri 2024 and Robbins 2023.

If purity ranking outperforms stability ranking in AUPRC → publishable finding.
If it does not → stability ranking (which has cleaner methodological precedent via StableMate) remains the primary method; purity ranking is a sensitivity/exploratory analysis.

---
type: condensate
depends_on:
  - "[[condensates/auprc-for-imbalance-handling]]"
  - "[[equations/brier-decomp]]"
applies_to:
  - "gates that combine discrimination and calibration into a single scalar utility under preference uncertainty"
  - "V3 imbalance axis-joint adjudication prior to stakeholder-elicited weight"
  - "contexts where decision-curve-analysis net-benefit over a clinically plausible threshold range has NOT been performed"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v3-imbalance
    delta: "0.5/0.5 weighting in U = 0.5 * AUPRC_norm + 0.5 * (1 - REL_norm) adopted per DECISION_TREE_AUDIT §2.3 resolution (2026-04-08) to correct the earlier raw-subtraction scale mismatch; no stakeholder-elicited alternative weighting was available"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §V3; v3-imbalance protocol §2.3, T-V3.4; DECISION_TREE_AUDIT.md §2.3"
falsifier: |
  If a documented downstream decision value (e.g., decision curve analysis
  net benefit across a clinically plausible threshold range, or an explicit
  stakeholder-elicited preference) consistently favors an alternative
  utility weighting (e.g., 0.7/0.3 or 0.3/0.7) by Direction claim against
  the 0.5/0.5 default across >=3 cohorts, the equal-weight default is
  weakened. If on >=5 cohorts, retire in favor of the elicited weighting
  rule.
---

# Equal weighting in V3 utility is the Laplacean default under preference uncertainty

## Claim

In V3's utility function

$$U = 0.5 \cdot \widetilde{\mathrm{AUPRC}} + 0.5 \cdot (1 - \widetilde{\mathrm{REL}}),$$

the equal 0.5/0.5 weighting of discrimination (AUPRC) and calibration
reliability (REL) reflects **two** stacking assumptions:

1. **Preference uncertainty** — no stakeholder-elicited relative
   preference between discrimination and calibration is available at V3
   decision time. Under preference uncertainty, equal weighting is the
   Laplacean prior: the choice that minimizes worst-case regret if either
   axis turns out to be the decision-relevant one.
2. **Structural decision relevance** — both metrics are load-bearing for
   V3's outputs. Calibration is locked downstream at V4, but REL at V3 is
   informative about which imbalance-handling cells produce well-ranked
   AND well-calibrated OOF posteriors (V4 then refines the calibration
   family). Locking an imbalance configuration that is purely
   discrimination-best but badly miscalibrated would force V4 to work
   harder and risks the per-fold-leakage path that
   [[condensates/calib-per-fold-leakage]] guards against.

The 0.5/0.5 default holds until **both** (a) an explicit stakeholder
preference is elicited or (b) a downstream decision value (e.g., DCA
net benefit over a clinically plausible threshold range) provides
evidence for an asymmetric weighting.

## Mechanism

The utility is a convex combination of two min-max-normalized metrics over
the 9 cells in the V3 grid. Both components are in `[0, 1]` post-
normalization — this is the scale-alignment fix from
`DECISION_TREE_AUDIT.md` §1.6 (the pre-2026-04-08 formulation subtracted
raw REL from raw AUPRC, mixing quantities with different natural scales).

Equal weighting corresponds to the Laplace principle of insufficient
reason: absent evidence for asymmetry, symmetric treatment is the move
that does not import an undocumented preference into a gate decision.
Under preference uncertainty, asymmetric weighting would be a silent
decision by the rulebook author about which axis the stakeholder would
prefer — precisely the kind of implicit threshold-shifting that
`DECISION_TREE_AUDIT.md` §1.2 flagged as "silent fallback" in the V3
context.

The equal-weight default is structurally falsifiable: any documented
stakeholder weighting or DCA-derived weighting becomes the new prior, and
V3's lock can be recomputed under the new weights as a sensitivity
analysis. Until then, equal weighting is the stable default that makes
the V3 lock auditable and reproducible across cohorts without cross-cohort
stakeholder interviews.

## Actionable rule

- V3's [[protocols/v3-imbalance]] uses **0.5/0.5** weighting in the
  utility function until a documented stakeholder preference or a DCA-
  derived weighting is available.
- V3's `observation.md` MUST log the **un-weighted** delta-AUPRC and
  delta-REL per cell alongside the weighted utility delta-U, so a
  retrospective re-weighting under an alternative rule (e.g., 0.7/0.3 or
  0.3/0.7) is reconstructable without re-running the experiment. This
  matches the T-V3.4 mandate in the v3-imbalance protocol.
- V3's `decision.md` MAY record an alternative weighting ONLY with
  explicit citation to (a) a stakeholder-elicited preference document, or
  (b) a DCA computation over a pre-declared threshold range. Any
  alternative weighting MUST re-declare the locks under both the 0.5/0.5
  default AND the alternative, so the sensitivity of the V3 lock to the
  weighting choice is documented.
- V3's `ledger.md` MUST cite this condensate when declaring the utility
  function, alongside [[condensates/auprc-for-imbalance-handling]] and
  [[equations/brier-decomp]].

## Boundary conditions

- Applies when V3 is being adjudicated before any stakeholder-elicited
  preference or DCA analysis is available. On a new cohort where a
  clinical DCA is available up front (e.g., a translational deployment),
  the elicited weighting supersedes the Laplacean default.
- Does NOT apply when utility is single-axis (e.g., pure discrimination
  for a non-clinical deployment where calibration is not used, or pure
  calibration for a pre-fitted model where discrimination is already
  locked). Single-axis adjudication does not use a weighted utility; it
  uses the relevant axis directly.
- Does NOT apply to post-V4 operating-point selection, where the
  threshold rule itself (fixed-spec, Youden, max F1) embeds a different
  preference structure via ADR-010.
- Does NOT apply to gates whose axes are purely architectural (V1
  recipe, V2 model) — those gates adjudicate on discrimination directly
  and use REL only as a secondary check.
- If both metrics are in strong Direction-level agreement (same cell
  dominant on both axes independently), the weighting is not load-
  bearing for the lock; equal-weight simply inherits that agreement. The
  weighting matters only when the two metrics disagree across cells —
  and those are exactly the cells where an elicited preference would
  have the most operational value.

## Evidence

| Dataset | Gate | Weighting | Rationale documented | Source |
|---|---|---|---|---|
| Celiac (UKBB) | v3-imbalance | 0.5/0.5 | No stakeholder preference elicited; no DCA net-benefit analysis performed at V3 decision time | DECISION_TREE_AUDIT.md §2.3 resolution 2026-04-08; v3-imbalance protocol §2.3 and T-V3.4 |

## Related

- [[condensates/v3-utility-provenance-chain]] — shared invariant this claim participates in
- [[protocols/v3-imbalance]] — protocol that cited this gap (Known
  tension T-V3.4 "0.5/0.5 utility weighting is a choice, not a derived
  rule"; §2.3 utility function)
- [[condensates/auprc-for-imbalance-handling]] — grounds the AUPRC half
  of the utility; this condensate grounds its weight
- [[equations/brier-decomp]] — grounds the REL half of the utility via
  the Brier decomposition; this condensate grounds its weight
- [[condensates/calib-per-fold-leakage]] — explains why REL is
  informative at V3 even though calibration-family is locked at V4 (V3
  must use OOF-posthoc REL to keep utility orthogonal to V4's family
  choice)
- DECISION_TREE_AUDIT.md §2.3 (cel-risk, 2026-04-08) — resolution
  record for the 0.5/0.5 weighting; ADR-008 and surrounding audit entries
  are the upstream reasoning

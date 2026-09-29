# TODO — Loss functions in DNNs (AG Hartig internship)

State as of 2026-09-29. Paste this into a new Claude chat to restore context.

## Project context (for a fresh Claude session)

- Master's internship, AG Hartig (Theoretical Ecology, Uni Regensburg), 6 weeks.
- Question: does residual-guided loss function selection improve DNN predictions,
  and how much does loss choice matter at all?
- Stack: R, `cito` (DNN), `DHARMa::createData` (simulation), `marginaleffects`,
  `pbmcapply` (parallel + progress), ggplot2.
- Setup: RStudio on uni server (20 of 24 cores) -> GitHub -> pulled to Pop!_OS PC.
- Design per iteration: simulated data, 80/20 train/test, 5 DNN losses
  (mse, mae, poisson, gaussian, nbinom) + bootstrapped GLM baseline,
  20 bootstrap replicates each, 100 iterations.
- My role: I write the code and make the decisions. Claude is a critic and
  advisor — debugging, literature appraisal, writing feedback. Not autonomous.

---

## 1. Code to-dos

### Critical — must fix before the Gaussian run

- [ ] **`preds_train` used in all four R2 formulas** (`script_gaussian.R`, lines
      ~104, ~110, ~163, ~170). The test R2 must use `preds` / `test`, the train R2
      `preds_train` / `train`. As written the GLM block crashes (`preds_train`
      does not exist yet) and R2 == R2_train everywhere, which kills the
      train/test convergence check.
- [ ] **`AME_true` is wrong for Gaussian.** Currently
      `true_effects * mean(train$true_mu)` — that is the Poisson/log-link chain
      rule. Identity link means `AME_true = true_effects`, full stop.
- [ ] **Delete the dead `mcfadden()` function** in the Gaussian script. It is
      never called and uses `dbinom`, which is wrong for continuous data.
- [ ] **Wrap each `dnn()` fit in `tryCatch`** so a failing loss returns an NA row
      instead of killing the whole iteration (see chat for the pattern).

### Diagnosis and verification

- [ ] Run `run_one_iteration(1)` **serially** to get a real error message —
      parallel errors from `pbmclapply` have no usable traceback.
- [ ] Confirm the suspected cause of "missing value where TRUE/FALSE needed":
      Poisson/NB loss on negative Gaussian targets -> NaN loss -> NaN in cito's
      early-stopping comparison.
- [ ] Always test with `iter <- 2` before launching a 100-iteration run.
- [ ] `install.packages("pbmcapply")` on the server if not done.

### Data-generating decisions

- [ ] Check empirically how many negatives the current settings produce:
      `min(sim$observedResponse)`, `mean(sim$observedResponse < 0)`.
- [ ] Scenario A: keep intercept = -1 (hard case, ~2/3 negative) and report
      Poisson/NB non-convergence as a result.
- [ ] Scenario B (optional): filter to positive values **for the Poisson/NB fits
      only**, and label it clearly as a different subsample. Do NOT filter for all
      models — truncation would bias every estimate and invalidate `AME_true`.
- [ ] Scenario C: intercept = 3, same effect sizes, all-positive Gaussian data.
      This is the clean test of link/variance mismatch.

### Open questions to verify

- [ ] How does cito combine `validation = 0.2` with `bootstrap = 20`? Hold-out
      before or inside each resample?
- [ ] Does `summary.citodnnBootstrap` really report mean / SD / z-test for ACEs?
      The bootstrapped GLM was built to mirror it — confirm they match.
- [ ] Mechanism for Poisson-loss degradation on Gaussian data: likely the
      Var = mu assumption misweighting observations, **not** Jensen's inequality
      (the network can learn eta = log(mu) directly). Ask Hartig.
- [ ] Consider `n_boot = 50` instead of 20 if compute allows — 20 gives roughly
      +-16% relative error on each SE, which propagates into coverage.

### Still to build

- [ ] `script_nbinom.R` — needs a dispersion parameter for its likelihood.
- [ ] Confirm `gaussian_coverage.csv` writes correctly (only tested for Poisson).

### Housekeeping

- [ ] Revoke the leaked GitHub PAT and generate a replacement. The blocked commit
      was removed from history, but rotate anyway.

---

## 2. Notes for the final report

### Methods — must be stated

- [ ] Simulated data via `DHARMa::createData`; `sampleSize = 250`, 80/20 split,
      `validation = 0.2` inside training, so each DNN fits on 160 rows.
- [ ] Parameter choice rationale: intercept = -1 and small effects
      (2, 0.4, 0.1) were chosen deliberately as a *hard* case. A higher intercept
      makes Poisson behave approximately Gaussian and erases the contrast.
- [ ] DNN predictions are a bagged ensemble over 20 bootstrap networks. The GLM
      baseline is bootstrapped identically (20 resamples, mean prediction,
      SD-based SE) so the comparison is like-for-like.
- [ ] Architecture: 2x50 ReLU, Adam, weight decay 0.01, lr 0.003, 200 epochs,
      early stopping patience 20.
- [ ] Weight decay and early stopping were added in response to earlier
      convergence problems.
- [ ] Goodness of fit: McFadden R2 (Poisson script) vs ordinary R2 (Gaussian
      script), keyed to the **data-generating process**, not the loss function.
      State explicitly that values are not comparable across scripts.
- [ ] Performance measures (coverage, bias, Monte Carlo SE) follow
      Morris, White & Crowther (2019), Stat Med 38(11).

### Results — findings worth reporting

- [ ] **Negative McFadden R2 for MAE loss on Poisson data.** Not an artefact:
      it means the model fits worse than the intercept-only null. Do not take
      `abs()`. Cross-reference with the `Neg_preds` column as the mechanism.
- [ ] **`Neg_preds` column**: how often each loss predicted impossible negative
      counts. Direct mechanistic evidence for loss/distribution mismatch.
- [ ] **Poisson and NB losses fail to converge entirely on Gaussian data
      containing negative values.** A practitioner would therefore never select
      them — the failure is itself the diagnostic signal.
- [ ] Coverage/bias table per loss x predictor. With 100 iterations the Monte
      Carlo SE on coverage is ~0.02, so 0.93 vs 0.95 is noise while 0.80 vs 0.95
      is real. Say this explicitly so the table is read correctly.
- [ ] Train/test boxplot -> appendix, as a model-validity check.

### Discussion

- [ ] Before `validation` was enabled, early stopping monitored the *training*
      loss and therefore did essentially nothing. Worth a sentence on why
      held-out monitoring matters.
- [ ] Error bars in the effects plots show the mean bootstrap SE (the model's
      self-reported uncertainty). Calibration — whether that SE is honest — is
      in the coverage table. Two different questions.

### Literature

- [ ] Cite `DHARMa` — Hartig, Methods in Ecology and Evolution.
- [ ] Cite `cito` — Runge & Hartig (check for the paper; otherwise cite the
      R package properly).
- [ ] Add Morris, White & Crowther (2019), Stat Med 38(11) — also a good
      template for structuring the Methods of a simulation study.
- [ ] Gneiting & Raftery (2007), JASA 102 — the proper-scoring-rules anchor
      linking loss functions and statistical estimation. Strong, use it.
- [ ] Guo et al. (2017), ICML — calibration of modern neural networks.
- [ ] Lathuiliere et al. (2020), IEEE TPAMI — top-tier, keep.
- [ ] LeCun, Bengio & Hinton (2015), Nature — keep for the intro.
- [ ] **Remove** the duplicate `li_human_2021` entry from the .bib.
- [ ] **Remove** the human-pose-estimation paper — wrong domain.
- [ ] Treat Ciampiconi et al. (MDPI *AI*) as background only; low-prestige venue.

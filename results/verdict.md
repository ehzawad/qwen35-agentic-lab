# Comparison verdict

ckpt             accuracy (95% CI)  calls/ep   runaway   no-box  tool-ok     n
------------------------------------------------------------------------------
base          0.810 [0.750, 0.858]       2.8     4/200   28/200    0.810   200
sft           0.050 [0.009, 0.236]      50.0     19/20    19/20    0.050    20
rssft         0.920 [0.874, 0.950]       1.4     0/200    4/200    0.840   200
rsgrpo        0.930 [0.886, 0.958]       1.2     0/200    4/200    0.930   200

## Harness sanity (must be clean before reading anything above)
  OK    base
  OK    sft
  OK    rssft
  OK    rsgrpo

rssft - base : +0.110  paired McNemar over 200 shared problems: b=5 c=27 z=+3.89 p=0.000
  unpaired (conservative): z=+3.22 p=0.001; unpaired MDE ~ 0.110 (paired design resolves less)
rsgrpo - rssft: +0.010  paired McNemar over 200 shared problems: b=7 c=9 z=+0.50 p=0.804
  unpaired (conservative): z=+0.38 p=0.704; unpaired MDE ~ 0.076 (paired design resolves less)

## Gates (registered before launch)
  PASS  G1 accuracy >= 0.800: rssft 0.920
  PASS  G2 calls/ep <= 6.0: rssft 1.4 (base 3.3, broken 50.0)
  PASS  G3 runaway <= 10%: 0/200
  PASS  G4 no-box < base: rssft 2.0% vs base 14.0%
  PASS  G5 rsgrpo >= rssft (directional): 0.930 vs 0.920

## Verdict: 5 passed, 0 failed, 0 skipped
Single-turn SFT destroyed termination; outcome-filtered multi-turn SFT restored it.

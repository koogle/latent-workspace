# Small-data convergence and layer controls

A window is 128 prefix tokens plus four future target tokens. An optimizer update
samples a batch of windows with replacement. Neither update counts nor reporting
intervals change the four-token generation block. Learning rate remains 0.0003;
the denser reporting schedule is not a learning-rate sweep.

The exploratory follow-up `small-layer-controls-01` uses nested 96/768/3072-window
subsets of the existing clean, balanced 196608-window training pool. Batch 32,
checkpoints 25/50/100/200/400/800/1600/3200 updates, all training windows measured
at every checkpoint. At 3200 updates the expected sampled passes are
1066.7/133.3/33.3 respectively; replacement sampling is not a literal epoch.
The same existing 192 validation windows are reused. The previously evaluated
test set is not evaluated again. This is an adaptive diagnostic, not confirmation.

Four otherwise matched diffusion heads attend all 24 residual layers: baseline,
25% whole-layer dropout, a null attention option, and both. Dropout is independent
per example and layer during training only. If every layer is dropped without a
null option, retain the final layer. The null option appends exactly zero K/V
vectors after attention projections (`MultiheadAttention.add_zero_attn`); its
logit is zero, so attention may allocate mass to no memory content. There is no
learned null embedding or routing regularizer. Workspace slot residual paths and
the common final-state output shortcut remain in every arm. Thus dropout does
not remove all final-state information.

Initial weights, ordered subsets, sampling and corruption RNG are matched.
Layer dropout uses a separate step-indexed RNG, reproducible under study resume.
Only selected training-prefix features are cached; the full frozen corpus and
ordered subset indices remain archived for auditing. All arms share one bounded
3300-second L40S job; incomplete arms must be reported as incomplete, not compared
as if they received equal training. Best checkpoints use validation NLL.

Interpretation: low training loss alone demonstrates fitting, not useful future
information. Train/validation gaps diagnose overfitting. Falling training loss at
the final checkpoint is evidence against claiming optimization convergence.
Small subsets cannot establish the final training-data requirement.

# Notes on Benchmarking


## General-purpose vs Task-specific algorithms

"General purpose" algorithms are treated differently from task-specific algorithms in Matbench for the purposes of ranking.

We make this distinction because some algorithms can be trained and used - in theory - for predicting **any property** of a material as long as they are trained on sufficient data.
Others are specialized for particular domains and need a separate comparison for fair analysis.


### General purpose algorithms 

General purpose algorithms are valid for all the tasks in a benchmark **using the same human-chosen configuration**.
Beyond defining a single configuration before beginning a benchmark, a human should not be hand-tuning or informing the algorithm about architecture, parameters, or hyperparameters.
However, general purpose algorithms can automatically determine hyperparameters and parameters as part of their fitting processes in each fold.


**We include algorithms as "general purpose" to include on the general purpose leaderboard if any one of the following criteria is met for Matbench v0.1:**

 - All 13 tasks are recorded, OR...
 - All 10 regression tasks are recorded, OR...
 - All 9 structure tasks are recorded. If only the 9 structure tasks are recorded, the algorithm is marked with "requires structure".


General purpose algorithms' results will appear on both the [General Purpose Leaderboard](../index.md) as well as the Task-specific leaderboards.


### Task specific algorithms

Task-specific algorithms can fit on any subset of tasks; for example, a single task. Task-specific algorithms may be valid or specialized only for a subset of the tasks in the benchmark.  

For example, if you have a model which was specifically created for predicting bulk metallic glasses, you may submit a benchmark containing only results for the `matbench_glass` dataset. 

Task-specific results **will only appear on the Task-specific leaderboards, not on the General Purpose Leaderboard**. 


---

## Why MAE?

Mean absolute error was chosen as the ranking metric for regression because:

1. The meaning of MAE is the most easily inferred
2. Dataset targets which should be analyzed according to relative error (such as bulk moduli) have their target transformed to order-of-magnitude form (e.g., log10).
3. MAE are valid for all target values, unlike mean absolute percentage errors, which are invalid for 0-valued targets.

That being said, other error metrics are also informative beyond what MAE can offer. Therefore, Matbench offers multiple error metrics to help assess generalization error.

---

## Mean absolute percentage error (`mape*`) scores
Mean absolute percentage error is only valid on sets of data without any true values of zero.
Also, small true values can result in very large MAPE for samples with even very small predicted error.
A threshold of 1e-5 is applied to mask samples with true absolute values smaller than this from the
MAPE calculation. The reported MAPE is the decimal (not percentage) among these masked samples; i.e., 
a MAPE of 11% corresponds to `mape*=0.11`, and a MAPE of 11,000% corresponds to `mape*=110`.

**Please use the given MAPE scores with a grain of salt, as they are not complete for the reason given above**.


---

## Classification metrics and historical ROC-AUC migration

Classification predictions should be a one-dimensional array of
positive/True-class probabilities. For classifiers with a two-column
`predict_proba` result, record `predict_proba(inputs)[:, 1]`. ROC-AUC uses these
original, unthresholded probabilities. Accuracy, balanced accuracy, and F1 use
boolean labels produced with the strict threshold `probability > 0.5`.

Older Matbench versions attempted to support both probabilities and labels, but
thresholded the prediction array in place while iterating over the
classification metrics. Because ROC-AUC was evaluated after the label-based
metrics, it received the thresholded 0/1 labels instead of the original
probabilities. This made stored ROC-AUC values label-based even when the result
file retained enough information to compute standard score-based ROC-AUC.

The historical migration audits all 27 submission-task records for
`matbench_glass`, `matbench_expt_is_metal`, and `matbench_mp_is_metal`. Eleven
records retained continuous probabilities and have been reproducibly rescored
from the canonical validation splits and prediction IDs. They occur in the
MODNet 0.1.10, MODNet 0.1.12, coGN, coNGN, DimeNet++, MegNet, and SchNet result
files. The migration preserves every other stored metric and historical field,
updates the result-file integrity hashes, and records per-fold scores, file
hashes, ranks, and scorer provenance in the committed manifest.

The remaining 16 records store boolean hard labels only. Their numeric
ROC-AUC values are unchanged because the original ranking scores cannot be
recovered. Leaderboards identify these as **hard boolean labels
(label-based/degenerate ROC-AUC)**. For binary hard labels this AUC is
equivalent to balanced accuracy, so it should not be interpreted as
score-based ROC-AUC or as an additional discrete AUC metric.





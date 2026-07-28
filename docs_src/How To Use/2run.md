
# 2 - Run benchmark on algorithm


## Recording your data

You can use the matbench python package to retrieve the training and testing splits as well as
record new predictions. Recording and saving your data with matbench should take no more than 
10 lines of matbench code.

The only things you need are:

1. Your algorithm/model - we'll call it `my_model` in this example
2. The `MatbenchBenchmark` class.


Here's an example of running an entire benchmark (13 tasks) using matbench.

```python
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    for fold in task.folds:

        # Inputs are either chemical compositions as strings
        # or crystal structures as pymatgen.Structure objects.
        # Outputs are either floats (regression tasks) or bools (classification tasks)
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        
        # train and validate your model
        my_model.train_and_validate(train_inputs, train_outputs)
        
        # Get testing data
        test_inputs = task.get_test_data(fold, include_target=False)
        
        # Predict on the testing data. Regression predictions are floats.
        # Classification predictions must be a one-dimensional array of
        # positive/True-class probabilities.
        if task.metadata.task_type == "classification":
            predictions = my_model.predict_proba(test_inputs)[:, 1]
        else:
            predictions = my_model.predict(test_inputs)
        
        # Record your data!
        task.record(fold, predictions)

# Save your results
mb.to_file("my_models_benchmark.json.gz")

```

**And you're done! Your benchmark has been recorded and saved.**

The output file, in this case `my_models_benchmark.json.gz` contains everything predicted by your 
benchmark. **Keep this file, as it is the core result that will be submitted to the leaderboard.**

For classification tasks, supply one positive/True-class probability per test
sample. In particular, scikit-learn-style `predict_proba` output has two
columns, so pass `predict_proba(test_inputs)[:, 1]`, not the full two-dimensional
array. ROC-AUC is calculated from these original probabilities. Accuracy,
balanced accuracy, and F1 are calculated from labels obtained with the strict
threshold `probability > 0.5`; a probability of exactly `0.5` is therefore
False. Non-one-dimensional classification inputs raise a `ValueError` rather
than being flattened silently.

Boolean hard-label predictions remain accepted for compatibility with
historical submissions. Their ROC-AUC is label-based and degenerate: for binary
labels it is equivalent to balanced accuracy and does not contain the ranking
information of score-based ROC-AUC. New submissions should always retain and
record positive-class probabilities.

Please see the docs for [Submitting to leaderboard](3submit.md) to learn how to upload your data to the automated leaderboard.


### Note: Benchmark subsets
If you want to benchmark on a subset of Matbench tasks, set the `subset` argument when creating `MatbenchBenchmark`
and use the same code as above. The repo accepts subsets of matbench tasks as well which will appear on a separate "task-specific" leaderboard.


---

## Recording hyperparameters and user metadata

### Hyperparameters for each fold

Record parameters (`dict` type) for each fold using the `parameters` argument to `MatbenchBenchmark.record`:

```python

from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        my_model.train_and_validate(train_inputs, train_outputs)
        test_inputs = task.get_test_data(fold, include_target=False)
        predictions = my_model.predict(test_inputs)
        
        # Get your model's parameters 
        # Parameters must be a dictionary of python native types, e.g., lists of strings, dicts, etc.
        params = my_model.get_parameters_as_dictionary()
        task.record(fold, predictions, params=params)

```

We recommend you record the hyperparameters on each fold - but it is optional. 

Your parameters can be freeform, though we encourage brevity - only recording the most important parameters, 
not large arrays or weight matrices.


### User metadata for benchmark

Add arbitrary metadata about your algorithm, in `dict` format, to the benchmark. This will be included as shown on the benchmark leaderboard on the website.

```python

my_metadata = {
    "algorithm_version": "v1",
    "tree_type": "entropy",
    "configuration": {
        "some_param": 4,
        "other_vector": [1, 2, 3]
    }
}

mb.add_metadata(my_metadata)
```






Table of Contents
=================

* [Install dependencies first](#install-dependencies-first)
* [Train the answerable solution (on the training set)](#train-the-answerable-solution-on-the-training-set)
* [Evaluate the answerable solution (on the validation set)](#evaluate-the-answerable-solution-on-the-validation-set)
   * [Reorder the prediction file](#reorder-the-prediction-file)
* [Further train the answerable solution (on the validation set)](#further-train-the-answerable-solution-on-the-validation-set)
* [Generate submission file](#generate-submission-file)
* [If you get differences](#if-you-get-differences)
* [Verbose Output](#verbose-output)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

# Install dependencies first

[install dependencies](../README.md)

# Train the answerable solution (on the training set)

```bash
env prediction_file=train.preds ref_metric=ansf1 ./qdecompdyn.py
```

I changed all the default settings to correspond to the submitted configuration, so this will:
* Use [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct).
* Use a rank-8 Lora Adapter.
* Use gated soft ref updating (`ref_metric=ansf1`)
   * If you want unconditional soft ref updating, don't set this environment variable (i.e., just run `./qdecompdyn.py`)
   * If you don't want soft ref updating at all, try `env ref_update_weight=0 ./qdecompdyn.py`
* Live subsample the training dataset to (approximately) match the hop moments with the dev set.
   * If you don't want this, try `env distribution_matching=False ./qdecompdyn.py`
   * To find more (undocumented) settings you can try, check out [Params.py](Params.py)
* Output running predictions to the file `train.preds` (`prediction_file=train.preds`)
   * This is progressive validation, i.e., predictions are made on an example before learning is done for that example.

This takes about 4 A100-days to run and will dump a bunch of checkpoints to the current directory.  To save time, you can use [snapshots/onetrainingpass](snapshots/onetrainingpass) which contains the final checkpoint from a single training pass on the training set.

# Evaluate the answerable solution (on the validation set)

If you ran training you can use a checkpoint from that.  Here I'll be using a checkpoint that has been included in the github repo.

```bash
env do_learning=False force_extractive=True prediction_file=val.preds split=validation final_model_id=snapshots/onetrainingpass/save_musique_qdecompdyn_final_final ./qdecompdyn.py
```

* `force_extractive=True`: This adjusts the output of the flow to correspond to a span in one of the selected documents.  Set to False if you don't want this.

This takes about 1 A100-day.

## Reorder the prediction file

If you have multiple GPUs than the predictions file can get reordered relative to the submission files, so you need to use the [reorder_preds_like.py](reorder_preds_like.py) script.

Assuming you have the [musique github repo](https://github.com/stonybrooknlp/musique) checked out in `~/musique`:
```bash
./reorder_preds_like.py val.preds ~/musique/data/musique_ans_v1.0_dev.jsonl > val.inorder.preds
```
or you can just call evaluate directly without creating an intermediate file
```bash
python ~/musique/evaluate_v1.0.py <(./reorder_preds_like.py val.preds ~/musique/data/musique_ans_v1.0_dev.jsonl) ~/musique/data/musique_ans_v1.0_dev.jsonl
```

If you ran the training step above, you can use this procedure with `train.preds` to compute a progressive validation evaluation.  It should almost exactly agree with the running display during training.

# Further train the answerable solution (on the validation set)

If you ran training you can use a checkpoint from that.  Here I'll be using a checkpoint that has been included in the github repo.
```bash
env dataset_seed=666 prediction_file=trainonval.preds ref_metric=ansf1 split=validation train_on_dev=True final_model_id=snapshots/onetrainingpass/save_musique_qdecompdyn_final_final ./qdecompdyn.py
```
This takes roughly 2 A100-days.  To save time, you can use [snapshots/onetrainingonvalpass](snapshots/onetrainingonvalpass) which contains the final checkpoint from a single training pass on the validation set.

# Generate submission file

If you ran training you can use a checkpoint from that. Here I'll be using a checkpoint that has been included in the github repo.
```bash
env do_learning=False force_extractive=True prediction_file=test.preds split=test final_model_id=snapshots/onetrainonvalpass/save_musique_qdecompdyn_final_final ./qdecompdyn.py
```
This takes about 1 A100-day.  Don't forget to reorder the predictions file before submitting.
```bash
./reorder_preds_like.py test.preds ~/musique/data/musique_ans_v1.0_test.jsonl > test.inorder.preds
```
Hopefully you end up with something that matches [phi.rank8.trainonval.forTrue.cpfinal.ans.test.preds](phi.rank8.trainonval.forTrue.cpfinal.ans.test.preds).

# If you get differences

I used the [1e10cf49da9eceb263824a4e4646d0ecba4f7dec](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/commit/1e10cf49da9eceb263824a4e4646d0ecba4f7dec) snapshot of Phi3.  Try to pin the model version and try again. 

# Verbose Output

If you set the `trace=True` environment variable, all the scripts will output verbose information about the internal processing of the flow.  For example
```bash
env CUDA_VISIBLE_DEVICES=0 trace=True do_learning=False force_extractive=True split=validation final_model_id=snapshots/onetrainingpass/save_musique_qdecompdyn_final_final ./qdecompdyn.py
```
will display information for a trained model on examples it was not trained on (the dev set) but for which you have gold labels available.

I included the `CUDA_VISIBLE_DEVICES=0` otherwise you'll get interleaved output from multiple worker threads.

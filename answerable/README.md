## Train the answerable solution (on the training set)

```bash
./qdecompdyn.py
```

I changed all the default settings to correspond to the submitted configuration, so this will:
* Use [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct).
* Use a rank-8 Lora Adapter.
* Use gated soft ref updating.
* Live subsample the training dataset to (approximately) match the hop moments with the dev set.

This takes about 4 A100-days to run and will dump a bunch of checkpoints to the current directory.

## Evaluate the answerable solution (on the validation set)

If you ran training you can use a checkpoint from that.  Here I'll be using a checkpoint that has been included in the github repo.

```bash
env do_learning=False force_extractive=True split=validation final_model_id=snapshots/onetrainingpass/save_musique_qdecompdyn_final_final ./qdecompdyn.py
```

This adjusts the output of the flow to correspond to a span in one of the selected documents (`force_extractive=True`).  You can set it to False if you don't want this.


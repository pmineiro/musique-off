## Train the answerable solution (on the training set)

```bash
./qdecompdyn.py
```

I changed all the default settings to correspond to the submitted configuration, so this will:
* Use [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct).
* Use a rank-8 Lora Adapter.
* Use gated soft ref updating.
* Live subsample the training dataset to (approximately) match the hop moments with the dev set.

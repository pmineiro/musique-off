# musique-off
Online Finetuned Flow Submission to Musique QA Benchmark

# QuickStart

## Install Dependencies

Starting in a checkout of the project.
```bash
conda create -y --name off python=3.10
conda activate off
pip install -r requirements.txt
```

## Train the answerable solution (on the training set)

```bash
cd answerable
./qdecompdyn.py
```

I changed all the default settings to correspond to the submitted configuration, so this will:
* Use (https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)[Phi-3].
* Use a rank-8 Lora Adapter.
* Use gated soft ref updating.
* Live subsample the training dataset to (approximately) match the hop moments with the dev set.

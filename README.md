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

## Train the answerable solution

I changed all the default settings to correspond to the submitted configuration.

```bash
cd answerable
./qdecompdyn.py
```

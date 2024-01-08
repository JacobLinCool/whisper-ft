# whisper-ft

Fine Tuning Whisper ASR.

## Setup

Use `env.yml` to create a virtual environment with all the dependencies.

```bash
micromamba env create -n whisper-ft -f env.yml
micromamba activate whisper-ft
```

> Use `conda`, `miniconda`, or `mamba` instead of `micromamba` if you prefer.

`HF_TOKEN` may be required to download and upload models/datasets from HuggingFace.

## Configuration

Check out the `config.py` file to see the available configuration options.

## Training

```bash
python ft.py
```

## References

- [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)
- [mozilla-foundation/common_voice_16_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0)

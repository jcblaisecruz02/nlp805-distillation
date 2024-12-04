# Multi to Monolingual Distillation
This repo is using code based on the [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) paper. We use the same base distillation data preprocessing and main KD code but **modified to perform layer copying**. Finetuning was performed via HuggingFace Transformers [examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) to ensure full reproducibility and compatibility with models uploaded to the Hub.

### Running the base experiments
1. Install all packages via `requirements.txt`
2. Copy the desired `config.json` file from `configs/` and rename to  `config.json` in the base folder.
3. Run the `tiny-distil` SLURM script (adjust paths to accommodate your setup) to pregenerate and run basic distillation.
4. Upload the resulting model the HuggingFace model hub
5. Run the `finetune` and `finetune-text` SLURM scripts to do the experiments. This requires the `examples/` folder from HuggingFace Transformers to be cloned.

### Notes
1. Full results table can be found on the paper.
2. Models are uploaded on HuggingFace hub (privated) instead of here due to size. Let me know if you need access.
3. The scripts will automatically download the datasets as needed. Otherwise, use `huggingface-cli` to download them to local and adjust path as necessary.

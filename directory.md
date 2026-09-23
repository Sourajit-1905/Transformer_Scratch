# Transformer from Scratch — Project Structure

```text
transformer_reproduction/         
│
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_data_preprocessing.ipynb       
│   ├── 03_attention.ipynb
│   ├── 04_multi_head_attention.ipynb
│   ├── 05_positional_encoding.ipynb
│   ├── 06_encoder.ipynb
│   ├── 07_decoder.ipynb
│   ├── 08_transformer.ipynb
│   ├── 09_training.ipynb
│   ├── 10_toy_overfit.ipynb
│   ├── 11_small_scale_experiment.ipynb
│   ├── 12_wmt14_experiment.ipynb
│   ├── 13_decoding.ipynb
│   ├── 14_bleu_evaluation.ipynb
│   ├── 15_ablation.ipynb
│   ├── 16_attention_visualization.ipynb
│   └── 17_final_results.ipynb
│
├── checkpoints/
│
├── datasets/
│   ├── stage_a/
│   ├── stage_b/
│   ├── stage_c/
│   └── stage_d/
│
├── vocab/
├── logs/
├── results/
├── translations/
│
├── figures/
│   └── attention/
│
├── configs/
│
├── experiments/
│   └── results.csv
│
├── data_pipeline.py               ← Saved utility module
├── tokenizer.py
├── attention.py  
├── multi_head_attention.py
├── positional_encoding.py
├── feed_forward.py
├── encoder.py
│
└── configs/dataset_config.json
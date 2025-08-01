# EAST: Embedding-Aligned Modality Adaptation for Large Language Model-Based Traffic Forecasting

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the official Pytorch implementation for our paper: "EAST: Embedding-Aligned Modality Adaptation for Large Language Model-Based Traffic Forecasting".

## Overview
Traditional deep learning methods for traffic forecasting typically rely on large amounts of localized training data, limiting their adaptability and generalizability across regions. Large Language Models (LLMs) are known for their prominent generalization ability, but they are trained on discrete, tokenized natural language, which creates a fundamental modality gap with continuous, multivariate traffic data. Existing LLM-based traffic forecasting methods often inject externally calculated embeddings into LLMs and extract the last hidden state to predict all future steps in a single pass. However, these methods do not address the modality gap problem and fail to exploit the token-by-token generation dynamics of LLMs. To address these limitations, we propose EAST (Embedding-Aligned Modality Adaptation for Large Language Model-Based Traffic Forecasting), a novel framework that exploits the use of visual representations for modality alignment and an auto-regressive LLM decoding strategy. Specifically, EAST transforms traffic flow sequences into image frames and extracts visual features using a video encoder, and captures sequential dependency features through a temporal encoder. It then aligns both features in a shared latent space via contrastive learning, producing semantically consistent spatio-temporal embeddings. Finally, the spatio-temporal embeddings are fed into an LLM to auto-regressively decode for multi-step traffic forecasting.

<p align="center">
  <img src="git/KG-Mber.jpg" alt="EAST model framework" width="900">
  <br>
  <b>Figure 1.</b> The model architecture of the proposed EAST framework.
</p>

## Datasets 

The dataset is located in `./dataset/NYC`.

## Quick-Start

Please run the following command to train the KG-Mber model on the Retail-Rocket dataset.

```bash
python train_prof.py 
```

## License

All data and code in this project are licensed under the [Apache License](./LICENSE).

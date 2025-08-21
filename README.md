# Vietnamese Student Feedback Sentiment Analysis using phoBERT and LoRA Fine-tuning

## Introduction

This project performs sentiment analysis on Vietnamese text, helping to identify positive, negative, or neutral sentiment from input sentences. The application can be used for product reviews, social media comments, and more.

Specifically, this project uses the [Vietnamese Students Feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) dataset from UIT NLP, which is provided free for research and non-commercial purposes.

## Project Purpose

The purpose of this project is to perform sentiment analysis on Vietnamese student feedback using advanced NLP techniques, specifically phoBERT and LoRA fine-tuning. By leveraging phoBERT—a pre-trained language model for Vietnamese—and applying LoRA (Low-Rank Adaptation) fine-tuning, the project aims to achieve high accuracy in classifying sentiment from text data.

### Why use LoRA Fine-tuning?

LoRA fine-tuning is chosen because it allows efficient adaptation of large pre-trained models like phoBERT with significantly fewer trainable parameters. This technique reduces computational resources and memory usage, making it possible to fine-tune powerful models even with limited hardware. LoRA also helps prevent overfitting and improves generalization, which is especially useful when working with domain-specific datasets such as student feedback.

## Features

- Text data preprocessing
- Sentiment analysis model training
- Sentiment prediction for new text
- Model accuracy reporting

## System Requirements

- Python 3.10+
- Libraries: scikit-learn, pandas, numpy, matplotlib, pytorch (install via pip)

## Installation

```bash
git clone https://github.com/<username>/sentiment-analysis.git
cd sentiment-analysis
```

## Usage

Train and predict sentiment:

```bash
# Train the model (usually done in the notebook)
jupyter notebook model.ipynb

# Or re-run the notebook via command line (if needed)
jupyter nbconvert --to notebook --execute model.ipynb
```

## Folder Structure

- `model.ipynb`: Notebook for training and evaluating the model
- `students_feedback_train.csv`, `students_feedback_test.csv`, `students_feedback_validation.csv`: Training, test, and validation datasets
- `results/`: Contains model checkpoints during training
  - `checkpoint-xxxx/`: Checkpoint folder for each training stage
    - `adapter_model.safetensors`: LoRA model weights
- `saved_model/`: Final saved model and evaluation results
  - `phobert-lora/`: Folder containing the fine-tuned LoRA model
  - `test_eval_metrics.json`: Evaluation results on the test set

## Dataset

- [Vietnamese Students Feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) from UIT NLP, free for research and non-commercial use.

## For More Information

For more information, please see the project slide:
[Project Slide](https://drive.google.com/file/d/1DZ2KNUXd0OWn6Qn_PsbCwS_s8LTP7N6a/view?usp=sharing)

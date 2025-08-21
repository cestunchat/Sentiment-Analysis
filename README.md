# Vietnamese Student Feedback Sentiment Analysis using phoBERT and LoRA Fine-tuning

## Introduction

This project performs sentiment analysis on Vietnamese text, helping to identify positive, negative, or neutral sentiment from input sentences. The application can be used for product reviews, social media comments, and more.

Specifically, this project uses the [Vietnamese Students Feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) dataset from UIT NLP, which is provided free for research and non-commercial purposes.

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
    - `README.md`: Model information at each checkpoint
- `saved_model/`: Final saved model and evaluation results
  - `phobert-lora/`: Folder containing the fine-tuned LoRA model
  - `test_eval_metrics.json`: Evaluation results on the test set
- `README.md`: Project documentation
- `.gitignore`: Git ignore configuration

## Dataset

- [Vietnamese Students Feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) from UIT NLP, free for research and non-commercial use.

## Contribution

All contributions are welcome! Please create a pull request or issue if you want to contribute or report bugs.

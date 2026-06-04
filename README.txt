
SENTIMENT ANALYSIS WITH RECURRENT NEURAL NETWORKS
Machine Learning and Big Data Processing - Project 2026

=======================================================

## PROJECT TITLE

Sentiment Analysis with Recurrent Neural Networks:
A Comparison of Vanilla RNN, GRU, and LSTM Models

## PROJECT DESCRIPTION

This project implements and compares three recurrent neural network architectures
for binary sentiment analysis on the IMDB Large Movie Review dataset:

1. Vanilla RNN
2. Gated Recurrent Unit (GRU)
3. Long Short-Term Memory (LSTM)

The task is to classify movie reviews as either positive or negative.

All recurrent cells are implemented manually from scratch using PyTorch tensor
operations and trainable parameters. The project does not use PyTorch's built-in
nn.RNN, nn.GRU, or nn.LSTM modules.

The models use pretrained GloVe word embeddings to initialize the embedding layer.
The default embedding file used by the training script is:

```
data/glove.6B.100d.txt
```

The submitted data folder also includes the 50d, 200d, and 300d GloVe files for
completeness, although the default experiment uses the 100-dimensional version.

The project compares the models in terms of:

* Test accuracy
* Validation accuracy
* Training time
* Convergence behavior
* Sensitivity to sequence length
* Performance on late-negation examples

## IMPORTANT IMPLEMENTATION NOTE

The recurrent models use true sequence lengths to avoid learning from padded
tokens. Padding tokens are assigned a zero embedding, and recurrent hidden states
are not updated after the real end of each review. This prevents padded timesteps
from corrupting the final hidden state used for classification.

## PROJECT STRUCTURE

ml_project_final/
├── README.txt
├── requirements.txt
├── run_full_training.py
├── comparison_results.png
├── training_output.log
├── rnn_model.pt
├── gru_model.pt
├── lstm_model.pt
├── src/
│   ├── data_loader.py
│   ├── late_negation_analysis.py
│   ├── models.py
│   ├── sentiment_net.py
│   └── train.py
└── data/
├── glove.6B.50d.txt
├── glove.6B.100d.txt
├── glove.6B.200d.txt
└── glove.6B.300d.txt

## SOURCE FILE DESCRIPTIONS

src/models.py
Contains from-scratch implementations of:
- RNNCell
- GRUCell
- LSTMCell

src/data_loader.py
Loads the IMDB dataset, builds the vocabulary, preprocesses text,
pads/truncates reviews, computes true sequence lengths, and loads GloVe
embeddings.

src/sentiment_net.py
Defines the full sentiment classification model:
embedding layer + recurrent layers + output classifier.

src/train.py
Contains training, validation, testing, sequence-length sensitivity analysis,
late-negation analysis, plotting, and model-saving logic.

src/late_negation_analysis.py
Contains helper functions for detecting reviews where negation words occur
near the end of the review.

run_full_training.py
Main script for training and comparing all three models.

requirements.txt
Python package dependencies.

comparison_results.png
Final comparison plot generated after training.

training_output.log
Full saved training log from the final experiment.

## REQUIREMENTS

Python 3.8 or newer is recommended.

Required Python packages:

* torch
* torchvision
* torchaudio
* datasets
* numpy
* matplotlib
* pandas
* scikit-learn
* nltk
* scipy
* tqdm
* gdown

## INSTALLATION

From the project root folder, install the dependencies with:

```
python3 -m pip install -r requirements.txt
```

If using a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## DATASET

This project uses the IMDB Large Movie Review dataset from Hugging Face.

The dataset contains:

* 25,000 training reviews
* 25,000 test reviews
* Binary labels:

  * 0 = negative
  * 1 = positive

The code automatically downloads the dataset on the first run using the
Hugging Face datasets library.

## PRETRAINED EMBEDDINGS

This project uses GloVe word embeddings.

The following GloVe files are included in the data folder:

```
data/glove.6B.50d.txt
data/glove.6B.100d.txt
data/glove.6B.200d.txt
data/glove.6B.300d.txt
```

The default training configuration uses:

```
data/glove.6B.100d.txt
```

If the GloVe 100d file is available at this path, it will be loaded automatically.
If the file is missing, the code can fall back to random embedding initialization,
but the submitted results were obtained using GloVe 100-dimensional embeddings.

## HOW TO RUN

To train and compare all three models, run:

```
python3 run_full_training.py
```

This will:

1. Load and preprocess the IMDB dataset
2. Build a vocabulary from the training split
3. Load GloVe embeddings
4. Train the RNN model
5. Train the GRU model
6. Train the LSTM model
7. Evaluate all models on the test set
8. Run late-negation analysis
9. Run sequence-length sensitivity analysis
10. Save the comparison plot as comparison_results.png
11. Save trained models as:

    * rnn_model.pt
    * gru_model.pt
    * lstm_model.pt

To save terminal output while training:

```
python3 run_full_training.py | tee training_output.log
```

## FINAL EXPERIMENTAL RESULTS

The final run was performed using:

* Dataset: IMDB Large Movie Review dataset
* Pretrained embeddings: GloVe 100d
* Vocabulary size: 10,000
* Embedding dimension: 100
* Hidden dimension: 128
* Number of recurrent layers: 2
* Dropout: 0.3
* Optimizer: Adam
* Learning rate: 0.001
* Epochs: 5
* Device: Apple MPS

Final results:

RNN:
Best Validation Accuracy: 78.88%
Test Accuracy: 78.16%
Test Loss: 0.4992
Training Time: 148.90 seconds
Final Training Loss: 0.4786
Late Negation Examples: 1000
Late Negation Accuracy: 80.90%

GRU:
Best Validation Accuracy: 88.28%
Test Accuracy: 87.42%
Test Loss: 0.3121
Training Time: 171.40 seconds
Final Training Loss: 0.1733
Late Negation Examples: 1000
Late Negation Accuracy: 93.60%

LSTM:
Best Validation Accuracy: 88.40%
Test Accuracy: 86.64%
Test Loss: 0.3160
Training Time: 183.99 seconds
Final Training Loss: 0.2051
Late Negation Examples: 1000
Late Negation Accuracy: 91.30%

## SEQUENCE LENGTH SENSITIVITY

Accuracy was evaluated using different maximum sequence lengths.

RNN:
50 tokens:  68.20%
100 tokens: 74.90%
200 tokens: 76.10%
300 tokens: 77.80%
500 tokens: 78.30%

GRU:
50 tokens:  78.00%
100 tokens: 83.50%
200 tokens: 87.30%
300 tokens: 88.80%
500 tokens: 89.20%

LSTM:
50 tokens:  78.70%
100 tokens: 82.10%
200 tokens: 85.60%
300 tokens: 87.40%
500 tokens: 87.50%

These results show that longer review context improves sentiment classification,
especially for the gated recurrent architectures.

## SUMMARY OF FINDINGS

The vanilla RNN achieved the lowest test accuracy but trained the fastest.
The GRU achieved the best test accuracy and the best late-negation accuracy.
The LSTM achieved the best validation accuracy but required the longest training
time.

The results demonstrate that gated recurrent architectures are more effective
than a vanilla RNN for sentiment analysis, especially when handling longer
reviews and late negation.

## OUTPUT FILES

After running the training script, the following output files are produced:

comparison_results.png
Plot comparing training loss, validation accuracy, sequence-length
sensitivity, and late-negation performance.

training_output.log
Full terminal output from the training run.

rnn_model.pt
Saved model weights for the best RNN model.

gru_model.pt
Saved model weights for the best GRU model.

lstm_model.pt
Saved model weights for the best LSTM model.

## NOTES FOR REPRODUCIBILITY

Random seeds are set in the training code for reproducibility. However, small
differences in results may occur depending on hardware, PyTorch version, and
device backend.

The code automatically uses the best available device in the following order:

1. CUDA GPU
2. Apple MPS
3. CPU

On CPU, training will be significantly slower.

## REFERENCES

Hochreiter, S., Schmidhuber, J.
"Long Short-Term Memory."
Neural Computation, 9(8), pp. 1735-1780, 1997.

Cho, K., van Merriënboer, B., Gulcehre, C., et al.
"Learning Phrase Representations using RNN Encoder-Decoder for Statistical
Machine Translation."
EMNLP, 2014.

Pennington, J., Socher, R., Manning, C. D.
"GloVe: Global Vectors for Word Representation."
EMNLP, 2014.

Maas, A. L., Daly, R. E., Pham, P. T., et al.
"Learning Word Vectors for Sentiment Analysis."
ACL, 2011.

## AUTHOR INFORMATION

Group Members: [Add group member names here]
Course: Machine Learning and Big Data Processing
Year: 2026



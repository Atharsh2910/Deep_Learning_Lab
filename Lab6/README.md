# Experiment 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

## Aim

The objective of this experiment is to build an end-to-end understanding of recurrent sequence
learning by implementing, training and comparing three recurrent architectures — Vanilla
RNN (SimpleRNN), LSTM and GRU — on a controlled, identical pipeline. The experiment also
covers Backpropagation Through Time (BPTT) and the vanishing/exploding gradient problem
that motivates gated architectures, extends the same idea to video action recognition using a
CNN feature extractor combined with a recurrent classifier, and finally studies the
encoder–decoder framework through a synthetic sequence-to-sequence (reversal) task.

## Dataset Description

**Primary dataset — UCI HAR (raw inertial signals):** The UCI *Human Activity Recognition
Using Smartphones* dataset contains smartphone accelerometer/gyroscope recordings for six
activities — WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
Each 128-time-step window carries 9 channels (3-axis body acceleration, 3-axis gyroscope,
3-axis total acceleration), giving `X ∈ R^(N×128×9)`. As recommended, a balanced laboratory
subset (~3000 windows) was drawn and split 70/15/15 into training (2100), validation (450)
and test (450) sets, with the same split and normalization statistics reused for every model.

**Video dataset — UCF101 subset:** Five real UCF101 action classes (Basketball, Biking,
WalkingWithDog, PlayingGuitar, JumpRope) were used, with 112 videos successfully processed
into 10-frame sequences.

**Sequence-to-sequence dataset:** A synthetic reversal task was generated — 5000 sequences
of length 4 with integers in [1, 9] (e.g. `[1,7,6,4] → [4,6,7,1]`), used purely to demonstrate
the encoder–decoder mechanism without needing an external dataset.

## Procedure

1. Load the raw UCI HAR inertial-signal files and arrange each window as `128 × 9`.
2. Draw a balanced subset, split 70/15/15, and normalize using training-set statistics only.
3. Visualize representative WALKING / SITTING / LAYING signals (Plot 1) to observe temporal patterns.
4. Manually compute BPTT hidden states for a toy 3-step example and verify against a program.
5. Build and train an identical classifier — `Input(128,9) → [SimpleRNN|LSTM|GRU](32) → Dropout(0.2) → Dense(16, ReLU) → Dense(6, Softmax)` — for each recurrent type, using Adam (lr=1e-3), batch size 32, 30 epochs.
6. Evaluate each model on the untouched test set: accuracy, macro precision/recall/F1, confusion matrix, parameter count, training time.
7. Repeat training at sequence lengths T ∈ {32, 64, 128} to study the effect of temporal context.
8. Extract frozen MobileNetV2 features (D = 1280) from 10 sampled frames per video, then train CNN-LSTM and CNN-GRU classifiers on the resulting `(N, 10, 1280)` tensors.
9. Build an encoder–decoder LSTM with teacher forcing for the reversal task and evaluate token/sequence accuracy.
10. Run seven additional exercises (unit sweep, GRU vs LSTM, stacked layers, bidirectional LSTM, sequence-length cost, video LSTM vs GRU, variable-length seq2seq).

## Techniques Used

- Vanilla RNN (SimpleRNN)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Backpropagation Through Time (BPTT)
- Transfer learning / frozen CNN feature extraction (MobileNetV2)
- CNN–LSTM / CNN–GRU hybrid architecture for video
- Encoder–decoder sequence-to-sequence learning with teacher forcing
- Bidirectional and stacked recurrent layers (additional exercises)

## Brief Description of Techniques

**Vanilla RNN** updates a hidden state `h_t = tanh(W_x x_t + W_h h_{t-1} + b_h)` at every
step, sharing weights across time but suffering from vanishing/exploding gradients over long
sequences because the same weight matrix is repeatedly multiplied during BPTT.

**LSTM** introduces a cell state and three gates (forget, input, output) that control what
information is discarded, written and exposed. Because the cell state updates additively
rather than multiplicatively, gradients flow more easily across long sequences.

**GRU** simplifies LSTM by merging the cell and hidden states and using only two gates
(update, reset), reducing parameters and computation while retaining most of the long-range
memory benefit.

**BPTT** unrolls the recurrent network across time steps into an equivalent feed-forward
graph and backpropagates the loss through every step to update the shared weights.

**CNN feature extraction (MobileNetV2)** uses a frozen, pretrained convolutional network with
global average pooling to turn each video frame into a 1280-dimensional spatial feature
vector, without training the CNN itself.

**Encoder–decoder (seq2seq)** compresses an input sequence into a fixed-size context via an
encoder LSTM, then generates the output sequence step-by-step in a decoder LSTM, using
teacher forcing (feeding true previous tokens) during training for stable, fast convergence.

## Results, Comparison and Evaluation Metrics

| Metric | RNN | LSTM | GRU |
|---|---|---|---|
| Accuracy (%) | 82.67 | 84.67 | **96.22** |
| Macro F1 (%) | 82.30 | 84.31 | **96.20** |
| Parameters | **1974** | 6006 | 4758 |
| Training Time (s) | 29.00 | 25.69 | **22.70** |

GRU clearly dominates on this dataset, achieving the highest accuracy/F1 with fewer
parameters and less training time than LSTM. The confusion matrices show the RNN and LSTM
confusing WALKING_UPSTAIRS/DOWNSTAIRS and SITTING/STANDING, while GRU almost perfectly
separates all six classes. Increasing sequence length generally helped LSTM/GRU but not the
plain RNN, confirming its weaker long-range memory. The CNN-LSTM/CNN-GRU video pipeline
reached 97.06% test accuracy on 5 UCF101 classes using 1280-D frozen MobileNetV2 features.
The seq2seq reversal task achieved 100% token and sequence accuracy after 40 epochs of
teacher-forced training.

## Summary

This experiment demonstrates why gated recurrent architectures outperform the Vanilla RNN:
LSTM and GRU's additive, gate-controlled memory updates resist vanishing gradients far better
than repeated tanh-weight multiplication, and on this HAR dataset GRU offered the best
trade-off between predictive performance, model complexity and training cost. Extending the
same recurrent modeling idea to video (via frozen CNN features) and to sequence generation
(via an encoder–decoder) confirmed that the core RNN/LSTM/GRU building blocks generalize
across sequence classification, video understanding and sequence transduction tasks alike.

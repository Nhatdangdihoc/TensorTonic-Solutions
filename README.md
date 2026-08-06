# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# B22DCKH081 - Trần Khánh Nhật's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/trankhanhnhat2k4.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Anchor Box Generation | Generate object-detection anchor boxes across a feature grid for every scale and aspect-ratio combination. | https://www.tensortonic.com/problems/anchor-box-generation |
| Compute Accuracy, Precision, Recall, F1 | Compute binary accuracy, precision, recall, and F1 score from predicted and true class labels. | https://www.tensortonic.com/problems/classification-metrics |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| ETL Schema Validation | Validate ETL records against required field names and data types, reporting rows that violate the schema. | https://www.tensortonic.com/problems/etl-schema-validation |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Compute Mean Average Precision (mAP) | Compute mean average precision across ranked retrieval results from per-query relevance labels. | https://www.tensortonic.com/problems/mean-average-precision |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| One-Step TD Value Update | Perform one temporal-difference value update from reward, discount, next-state value, and learning rate. | https://www.tensortonic.com/problems/td-value-update |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Coarse-Grained MoE Feedforward Block | Implement Arcee Trinity's coarse-grained MoE feed-forward block with top-k routing and expert output aggregation. | https://www.tensortonic.com/research/arcee-trinity/at-coarse-moe |
| Interleaved RoPE + NoPE Layer Pattern | Route Arcee Trinity layers through an interleaved pattern of rotary-position and NoPE attention. | https://www.tensortonic.com/research/arcee-trinity/at-interleaved |
| Segment Embeddings | Build BERT input embeddings by summing learned token, position, and sentence-segment embedding vectors. | https://www.tensortonic.com/research/bert/bert-segment-embedding |
| WordPiece Tokenization | Implement BERT WordPiece tokenization with greedy longest-match subwords, continuation prefixes, and unknown-token fallback. | https://www.tensortonic.com/research/bert/bert-wordpiece |
| KV Compression via Low-Rank Down-Projection | Implement DeepSeek V3 KV compression by projecting hidden states into a shared low-rank attention latent. | https://www.tensortonic.com/research/deepseekv3/ds3-kv-compress |
| KV Reconstruction via Up-Projection | Reconstruct DeepSeek V3 key and value representations by up-projecting the compressed KV latent for each head. | https://www.tensortonic.com/research/deepseekv3/ds3-kv-reconstruct |
| GAN Discriminator | Implement a GAN discriminator that maps input samples through dense layers to real-versus-fake probabilities. | https://www.tensortonic.com/research/gan/gan-discriminator |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| GAN Loss Functions | Compute numerically stable binary cross-entropy losses for the GAN generator and discriminator objectives. | https://www.tensortonic.com/research/gan/gan-loss |
| RMSNorm | Implement GLM-4.5 RMSNorm by scaling hidden states with inverse root-mean-square magnitude and learned weights. | https://www.tensortonic.com/research/glm45/glm-rmsnorm |
| Position-wise Feed-Forward Network | Implement the GPT-2 position-wise feed-forward network with expansion, GELU activation, and output projection. | https://www.tensortonic.com/research/gpt2/gpt2-ffn |
| MXFP4 Routed Expert Linear | Implement Kimi K3 MXFP4 expert linear layers by decoding packed E2M1 values and E8M0 block scales before projection. | https://www.tensortonic.com/research/kimik3/k3-mxfp4-expert-linear |
| Complete LSTM Cell | Build a complete LSTM cell with forget, input, candidate, cell-state, output, and hidden-state calculations. | https://www.tensortonic.com/research/lstm/lstm-cell |
| Cell State Update | Implement the LSTM cell-state update by combining retained memory with input-gated candidate information. | https://www.tensortonic.com/research/lstm/lstm-cell-state |
| Forget Gate | Implement an LSTM forget gate by combining the previous hidden state and current input with a sigmoid projection. | https://www.tensortonic.com/research/lstm/lstm-forget-gate |
| Complete LSTM Network | Assemble an LSTM sequence forward pass that carries hidden and cell states across every time step. | https://www.tensortonic.com/research/lstm/lstm-full-network |
| Input Gate | Implement the LSTM input gate and candidate activation that control new information written to the cell state. | https://www.tensortonic.com/research/lstm/lstm-input-gate |
| Output Gate | Implement the LSTM output gate and expose the current hidden state from the updated cell memory. | https://www.tensortonic.com/research/lstm/lstm-output-gate |
| Backpropagation Through Time | Implement one backpropagation-through-time step using the tanh derivative and hidden-to-hidden weight gradients. | https://www.tensortonic.com/research/rnn/rnn-bptt |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Complete Vanilla RNN | Assemble a vanilla RNN that processes sequences into recurrent hidden states and per-time-step output logits. | https://www.tensortonic.com/research/rnn/rnn-full-network |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/trankhanhnhat2k4)
<!-- tensortonic:end -->

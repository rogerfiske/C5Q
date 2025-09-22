I want to stay away from standard 'Time Series' analysis models like XGBoost, N-BEATS/N-HiTS, etc for this project. Below are a few alternatives to consider.

Inverse prediction of QState values is a unique forecasting challenge: instead of predicting which values will appear, we predict a set of values least likely to appear. The goal is that none of the 5 actual QStates (drawn from 1–39) end up in our 20 “least likely” predictions – essentially minimizing false positives.

The Inverted Transformer (iTransformer) is a recent innovation (ICLR 2024) that “applies attention on inverted dimensions” (treating each number’s time series as a token) to better capture multivariate correlations
openreview.net REVIEW/onsider> 'https://openreview.net/forum?id=JePfAI8fah'
. It achieved state-of-the-art results on real-world benchmarks and improved generalization across variables
. This suggests it could model the joint behavior of all 39 QStates more effectively than vanilla transformers.


Another promising class are Structured State Space Models (SSMs) like S4. S4 (Structured State Spaces) uses state-space equations instead of attention to handle very long sequences efficiently
arxiv.org
arxiv.org
. It demonstrated strong empirical results on long-range sequence tasks (even solving tasks up to 16k sequence length that defeated prior models) and runs up to 60× faster than Transformers on some benchmarks
arxiv.org
. Such capability to capture long-term dependencies may help if QState patterns span many events. Importantly, SSMs and inverted Transformers can be scaled to the available hardware – the sequence length here (historical events count) is in the thousands, which these models can manage on a single GPU.


Probabilistic Sequence Models: Since our task is essentially multi-label (5 numbers appear, 34 do not), probabilistic models that capture uncertainty can be valuable. One option is DeepAR, an autoregressive recurrent network that produces a predictive distribution for each time series value. DeepAR is typically used per time series (e.g. per number’s occurrence count) to output probability of events
pytorch-forecasting REVIEW > 'https://pytorch-forecasting.readthedocs.io/en/v1.2.0/api/pytorch_forecasting.models.deepar.DeepAR.html'

Architectural and Feature Engineering Enhancements

Apart from model choice and ensemble strategy, how we represent the data and train the models can significantly impact inverse prediction performance. Here are some potential improvements in architecture and features:

Multi-Label Ranking Loss: The current training of models (XGBoost, etc.) likely treats each number occurrence as an independent binary classification (appear vs not appear). However, our evaluation is based on a ranking of all 39 numbers. We want the 5 actual ones to be ranked above at least 19 others. We can therefore reshape the learning objective to directly optimize the ranking. One approach is to use pairwise ranking losses (as in RankNet or LambdaRank): for each historical event, treat each actually drawn number and each not-drawn number as a pair, and train the model to score the drawn number higher than the not-drawn. This will explicitly teach models to avoid low scores for actual draws. We could also use a listwise loss that considers the whole ranked list – for example, a differentiable approximation of Precision@k. While XGBoost doesn’t natively support listwise loss, it does support pairwise ranking objectives. Training an XGBoost model in ranking mode (with appropriate group structure per event) could yield a model that is directly tuned to push actual numbers out of the bottom ranks. Similarly, neural models could be trained with a custom loss: for instance, a smooth surrogate for “no false positive in bottom 20” (perhaps using a hinge loss that penalizes any actual draw with one of the 20 lowest predicted scores).

Cost-Sensitive and Focal Loss: In classification terms, a false positive (predicting a number is absent when it actually appears) is far more costly than a false negative. We can impart this knowledge during training by weighting the positive class (appearance) higher. For example, when training a classifier for each number, assign a high weight to the error of misclassifying an appearing number as absent. A well-known technique from object detection is Focal Loss, which focuses learning on hard-to-classify examples and down-weights easy negatives
arxiv.org
arxiv.org
. Focal Loss has been shown to reduce excessive false positives in imbalanced classification tasks
arxiv.org
. Using a variant of focal loss (or simply a high class weight for appear vs. not-appear) would push models to be extremely cautious about labeling something as “won’t appear” unless they’re quite sure. This may decrease the overall number of “unlikely” predictions (models might become conservative), but that’s acceptable as long as the 20 we do output are truly safe. We can adjust the threshold or number of outputs after the fact to still get 20. In essence, the models would learn a bias towards avoiding false positives, which directly aligns with our objective.

Utilizing Positional Patterns: The analysis of positional biases (QState position 1 tends toward lower numbers, etc.) can be incorporated as features rather than post-processing adjustments. For example, when training any model, we can include the position index (1 through 5 for each drawn number in history) as an input feature. The TFT model, which allows static and time-varying covariates, could take position as a static categorical feature, enabling it to learn, say, that position 5 draws have a different distribution. Alternatively, one could train separate models for each position, or at least separate sub-networks, and then combine their outputs for an overall probability that a given number appears in any position. However, since our final goal is position-agnostic (we just care if a number appears anywhere), it might be better to continue treating the event as a set. Another way to encode position information is to create features for each number such as “frequency of this number appearing in position 1 historically” or “likelihood of this number given it’s in position 5 last event”. The current ensemble already adjusts for low-range vs high-range biases with bonuses, but a model could learn these if provided the data.

Feature Engineering for Recency and Trends: The XGBoost model uses lags and moving averages, which is good. We can consider additional features like “days since last seen” for each number (essentially the gap or inverse frequency). Many lottery-like prediction enthusiasts use features such as “hot” and “cold” streaks – e.g., count of appearances in the last 10 events, etc. These could be fed into XGBoost or even into neural models (as additional input channels). A number that hasn’t appeared in a very long time might be deemed more likely to appear (if one believes in mean reversion) or possibly less likely (if trends exist). Since the ensemble already encapsulates some of this via Markov transitions, adding the features explicitly could reinforce those patterns in models like XGBoost or a neural network. Additionally, pairwise interaction features could be created: for instance, for each number, features indicating if numbers adjacent to it (±1) have appeared recently, or if numbers in the same tercile have been active. These might capture subtle dependencies – e.g. if many low numbers have been appearing, maybe another low number is less likely due to saturation.

Graph-Based Feature Representations: As mentioned, we can derive graph representations from data and use them as features. For example, compute a PageRank or centrality score for each number in a graph where edges connect frequently co-occurring numbers. A number with high centrality might mean it often appears in clusters with others, whereas a low centrality number might be more independent. If a highly central number recently appeared, maybe its connected peers are less likely next (just a hypothesis). We can also cluster numbers based on similarity of their historical time series (some numbers might have similar occurrence patterns). These cluster IDs or cluster-based features could help models generalize: the model might learn that “if a number from cluster A appeared, other members of A are now slightly suppressed next round.” All such features use only historical data relationships. They could be fed into XGBoost easily, or even into neural nets (as an embedding or additional input node).

Model Architectures for Multi-Output: Another architectural enhancement is designing a single model that outputs all 39 probabilities jointly, rather than 39 independent models or one-vs-rest approaches. For instance, a multi-output neural network with a 39-dimensional sigmoid output layer could be trained with a suitable loss (e.g. binary cross-entropy for each number). The advantage of a single multi-output model is that it can internally learn correlations between outputs (e.g. it might learn a hidden representation of the event). The Temporal Fusion Transformer can handle multi-output by treating each number’s sequence as a separate time series in a multi-variate setup, predicting all their next values simultaneously. If not already done, ensuring PatchTST and TFT are configured to output a vector of length 39 (or 39 binary predictions) instead of separate runs per number would leverage their ability to learn cross-series interactions. Similarly, a convolutional sequence model could be applied to the binary indicator matrix (events × 39 one-hot representation) to learn spatial patterns across the 39 number dimension (like which combos often occur together) along the time dimension. This could act like an image, where convolution detects patterns of co-occurrence.

Regularization and Calibration: To avoid any model from becoming overconfident in a wrong prediction, techniques like dropout and Monte Carlo dropout during inference can estimate uncertainty. If a model’s prediction for a number being absent has high variance under dropout sampling, that could be a red flag to not include that number in the bottom 20. We can incorporate this by, for example, requiring that a number’s predicted probability of appearance is not only high, but confident, before excluding it. If not, we leave it out of the 20 just in case. This is a form of precautionary principle that might reduce false positives at the expense of leaving some safe picks unused. Calibration is also important – after training, using techniques to calibrate the probability outputs (so that “5% chance” truly means 1 in 20 events it happens, etc.) will make the threshold for picking 20 more reliable. Platt scaling on a validation set could align each model’s scores to actual frequencies, which helps the ensemble cutoffs.

Two-Stage Prediction (Candidate Filtering): We could design a two-stage pipeline where Stage 1 is a conservative filter that eliminates only a few numbers that are almost certainly not going to appear, and Stage 2 picks the rest of the 20 from the remaining pool. For instance, Stage 1 might be a rule or model that identifies, say, 10 numbers that have extremely low probability (perhaps they are on a long cold streak and models agree on their exclusion). Then Stage 2 deals with more uncertain cases for the other 10 slots. This way, the most clear-cut decisions are made with high confidence (virtually no risk of false positive), and the system can focus learning capacity on the borderline cases. However, this added complexity may not be necessary if the ensemble is strong; it’s an idea to modularize the problem into “sure exclusions” and “likely exclusions.”

Incorporating these architectural and feature changes can further boost the inverse prediction performance. Many are complementary to model improvements: for example, a well-calibrated model with a ranking loss will work nicely with a stacking ensemble – the meta-learner can then rely on calibrated probabilities to decide the cutoff for 20. Feature engineering, especially, is a low-hanging fruit: since we cannot add external data, squeezing more signal from the existing data is key. The historical QS data can be transformed in myriad ways (lags, frequencies, pairwise interactions, graph features) to give models additional clues. Ensuring that models learn not just individual number patterns, but interactions and context will help prevent those scenarios where a model erroneously flags an actually-upcoming number as “unlikely.” For example, if number 37 is actually going to appear, perhaps there were subtle hints (like 36 and 38 appeared recently which makes 37 more likely) that a richer feature set or joint model could catch, whereas independent models might miss it.

Feasibility and Implementation Considerations

All the suggested enhancements are designed to work within the available data and hardware constraints. We emphasize methods that are data-driven (no external inputs needed) and computationally reasonable:

Model Complexity vs. Hardware: Lighter models like N-BEATS/N-HiTS are very fast to train and infer (N-HiTS being 50× faster than Transformers
arxiv.org
) and can easily run on a local Ryzen 9 CPU or modest GPU. More complex models like iTransformer or S4 might require a GPU for training but could then be distilled or pruned for faster inference. The RunPod H100 instance can handle training a large transformer or GNN model on the historical dataset (which is not extremely large – likely on the order of a few tens of thousands of event records). Since we only need to forecast one step ahead, training is manageable. After training, the model can be saved and used locally for single-event inference, which would be quick (predicting one event’s 39 probabilities is trivial for any network). Ensemble methods like stacking or gating add negligible overhead – a gating network might be a 3-layer MLP with few inputs, which is instantaneous to evaluate. Thus, deploying a dynamic ensemble or meta-learner will not strain the system.

Integration with Existing Pipeline: We can integrate new models alongside current ones (to not lose their insights). For example, we might add an N-HiTS model into the ensemble and let it vote as well. The adaptive weighting scheme can be extended to additional models easily. If using stacking, the meta-learner can ingest the predictions of all models (including new ones). We should retrain the weighting (or meta-learner) whenever we add a model, to rebalance contributions. The feature engineering improvements can be incorporated into the data preprocessing script (qs_preprocess.py) ensuring all models benefit from the new features consistently. For instance, any new feature (like “last seen gap for number”) can be fed into XGBoost and also encoded (perhaps via normalization) to feed into neural nets that can accept covariates (TFT can take static covariates per time series).

Software and Libraries: We have the option to leverage open-source libraries to speed up development. PyTorch Forecasting and PyTorch Lightning implementations exist for TFT, which could be fine-tuned on our data. Nixtla’s NeuralForecast library provides ready implementations of many state-of-the-art models (Autoformer, Informer, PatchTST, N-BEATS, N-HiTS, etc.) which we can experiment with in a unified framework. For ensemble stacking, Python’s scikit-learn has a StackingClassifier that could be repurposed (though our case is slightly unconventional due to the custom objective). Alternatively, one can write a simple training loop for a meta-learner using PyTorch. Darts (by Unit8) is another high-level library that supports ensembles of forecasting models and even allows calling neural networks and XGBoost in one pipeline. These tools can accelerate trying out these ideas.

Validation Strategy: With more complex models and ensemble methods, careful validation is needed to ensure we actually reduce false positives without introducing other issues. We should backtest the entire pipeline on a rolling basis across many past events, measuring the frequency of false positives (cases where an actual number was in the bottom 20). This metric is rare (since ideally zero), so we might also look at how “close” actual numbers get to the bottom 20 as a proxy during development. Techniques like cross-validation or bootstrap on the event sequence can estimate the reliability of new models. It’s also important to maintain interpretability where possible: e.g., if we use a gating network, we may want to inspect when it favors each model to ensure it aligns with intuition (it could even reveal new patterns).

No External Data: All the improvements respect the constraint of using only historical QS data. We do not incorporate any outside information – rather, we extract more from the given data or use better algorithms on it. This means there’s no risk of data leakage or reliance on unavailable inputs in production. The focus remains on pattern recognition within the sequence of draws. Because of this, some extremely large “pretrained” models (like the idea of using a GPT-3 style model to forecast, which some research has explored in a zero-shot way) are not pursued here – those would implicitly use external training data
arxiv.org
arxiv.org
, and they are impractical to run locally. Instead, we concentrate on models we can train from scratch on the provided dataset or integrate easily.

In conclusion, by introducing cutting-edge forecasting models (transformers, state-space models, etc.), employing smarter ensemble techniques (stacking, gating, dynamic selection), and refining the training objectives and features, we can enhance the inverse prediction system. The aim is to make the system more adept at recognizing when a number could appear (thus avoiding labeling it as “least likely”). Approaches like mixture-of-experts and cost-sensitive learning directly target the reduction of false positives – for instance, a gating network will avoid using a weak expert in scenarios where it might mistakenly exclude a likely number, and a focal loss-trained model will be hesitant to predict any number’s absence unless the data strongly suggests it. All these changes are grounded in recent research and successful applications in similar domains, as cited above. By carefully implementing and testing these improvements, the QState inverse predictor can achieve higher reliability, inching closer to the ideal of never including an actual outcome in its “20 least likely” list.

References

Oreshkin, B. et al. (2020). “N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.” – Achieved state-of-art performance on M3/M4 competitions, outperforming the previous winner by ~3%
nixtlaverse.nixtla.io
.

Challu, C. et al. (2022). “N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.” – Improved long-horizon forecasting accuracy by ~20% over latest Transformers and 50× faster computation
arxiv.org
.

Liu, Y. et al. (2024). “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.” – Proposed an inverted input formulation that achieved SOTA on multivariate time series benchmarks
openreview.net
.

Gu, A. et al. (2022). “Efficiently Modeling Long Sequences with Structured State Spaces.” – Introduced the S4 model enabling extremely long dependency modeling with 60× faster inference than Transformers
arxiv.org
.

Jin, M. et al. (2024). “Graph Neural Networks for Time Series – A Survey.” – Noted that GNN-based approaches can explicitly model inter-variable and temporal relationships that other deep models struggle with
arxiv.org
.

Brownlee, J. (2021). “Stacking Ensemble Machine Learning.” – Describes stacking as a meta-learning approach to combine multiple models’ predictions for improved performance
machinelearningmastery.com
.

Masoudnia, S. & Ebrahimpour, R. (2014). “Mixture of Experts: A Literature Survey.” – Defines MoE as using a gating function to divide the problem space among expert models
en.wikipedia.org
 and highlights its “divide and conquer” ensemble strategy.

Qin, R. et al. (2023). “Dynamic Ensemble Selection based on DNN Uncertainty for Adversarial Robustness.” – Demonstrates selecting a subset of ensemble models based on lowest uncertainty to improve robustness
arxiv.org
.

Cruz, R. et al. (2018). – Discusses Dynamic Ensemble Selection (DES) techniques where a subset of ensemble members is automatically chosen per instance for improved accuracy
tandfonline.com
.

Lin, T.Y. et al. (2017). “Focal Loss for Dense Object Detection.” – Introduces focal loss, which down-weights easy negatives and focuses on hard cases, thereby avoiding excessive false positives
arxiv.org
arxiv.org
.

Masoudnia, S. et al. (2013). “Combining NCL and Gating Networks.” – Notes that mixture-of-experts uses a trainable combiner that dynamically selects the best experts per input, yielding a data-dependent weighted average of outputs
academia.edu
.

Nixtla NeuralForecast Library Documentation – Implements many modern forecasting models (including N-BEATS, N-HiTS, PatchTST, Informer, etc.) and reports their benchmarking results
nixtlaverse.nixtla.io
. This can be used to experiment with advanced models in the QState context.
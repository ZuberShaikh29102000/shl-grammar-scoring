# Grammar Scoring Engine for Spoken Audio

**SHL Hiring Assessment 2026**

This model predicts a grammar score from 0 to 5 for spoken English audio clips. It transcribes each clip with two speech recognition models, extracts features from the text and from the audio, and blends three models into one score.

| Metric | Value |
|---|---|
| Training RMSE (in-sample) | **0.4215** |
| Out-of-fold RMSE (honest estimate) | **0.5066** |
| Out-of-fold Pearson | **0.9128** |
| Baseline RMSE (always predict the mean) | 1.2382 |

---

## Pipeline

```
          +-- Whisper      -> fluent, punctuated transcript --+-- linguistic features --+
audio ----+-- wav2vec2 CTC -> verbatim transcript ------------+-- DeBERTa regressor ----+--> blend --> calibrate --> score
          +-- WavLM        -> all-layer mean+std embedding -------------------------------+
```

1. **Transcription**
   - `faster-whisper` (small) gives a fluent, punctuated transcript.
   - `wav2vec2-large-960h-lv60-self` (greedy CTC) gives a near-verbatim transcript that keeps speaker errors.
2. **Linguistic features (37):** error density (LanguageTool), syntactic complexity (spaCy), fluency and disfluency, and language-model perplexity, computed on both transcripts.
3. **Acoustic embeddings:** `microsoft/wavlm-base-plus`, with every hidden layer pooled by mean and standard deviation.
4. **Text model:** `DeBERTa-v3` fine-tuned as a regressor on the Whisper transcript and averaged over several seeds.
5. **Blending and calibration:** blend weights and a linear calibration are fit on out-of-fold predictions, and scores are clipped to the range [0, 5].

## Validation

- 5-fold cross-validation, repeated with 4 seeds, reported as mean ± std
- Folds stratified on binned labels
- A change is kept only if it improves the mean by more than the fold-to-fold std

### Component results (out-of-fold)

| Model | RMSE | Pearson |
|---|---|---|
| Constant baseline | 1.2383 | -0.016 |
| Ridge on linguistic features | 0.7621 | 0.788 |
| GBM on linguistic features | 0.7160 | 0.816 |
| DeBERTa-v3 (7 seeds) | 0.7462 | 0.801 |
| WavLM, final layer, mean only | 0.6224 | 0.865 |
| WavLM, all layers, mean+std | 0.5674 | 0.889 |
| **Calibrated blend** | **0.5066** | **0.913** |

## Key findings

- **The CTC transcript's advantage is not about grammar.** CTC features beat Whisper features, but the gain came from word-count measures. The CTC-to-Whisper word-count ratio alone reaches Pearson 0.57. That ratio measures how hard the speech is to recognise, not grammatical quality.
- **Most of the prediction comes from acoustics.** The audio-only WavLM model, which never sees a word, gets about 68% of the blend weight.
- **Speaker leakage likely inflates the out-of-fold scores.** Grouping folds by pseudo-speaker clusters makes RMSE noticeably worse, so the out-of-fold numbers above are probably optimistic.

## Limitations

A large part of the model's accuracy comes from how speakers *sound* (accent, fluency, voice quality) rather than from their grammar. For a real assessment system this is a fairness risk: speakers with sound grammar but harder-to-recognise speech may be scored unfairly. Checking this properly needs speaker metadata that this dataset does not include.

## Running the notebook

The notebook is built for **Kaggle**:

1. Add the `shl-hiring-assessment-2026` competition as a data source.
2. Set the accelerator to **GPU T4 x2** and turn **Internet On**.
3. Click **Save Version → Save & Run All**. A full run takes about 2–3 hours.

To run it locally, install the dependencies and change `DATA_DIR` / `OUT_DIR` in section 1:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`language_tool_python` also needs Java installed.

## Repository structure

```
.
├── grammar_scoring_engine.ipynb   # full pipeline, analysis and submission
├── requirements.txt
└── README.md
```

The dataset is not included. Get it from the Kaggle competition page.

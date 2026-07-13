# koroku-tamil

Training recipe and Google Colab notebook for fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) on Tamil (தமிழ்). 

This repository is adapted from [shreyaskarnik/bol-tts-marathi](https://github.com/shreyaskarnik/bol-tts-marathi) and optimized to load speech training data directly from Hugging Face Parquet shards using **DuckDB SQL queries**.

## What This Is
- **SQL-Based Training Dataset Loader**: Bypasses the need to extract tens of thousands of individual `.wav` files to disk or Google Drive. Parquet audio bytes are lazily fetched by index and decoded in memory using DuckDB.
- **Tamil G2P Integration**: Dynamically phonemizes Tamil text using `misaki.espeak.EspeakG2P(language="ta")` during data loading.
- **Google Colab Training Notebook**: A jupyter notebook (`koroku_tamil_training.ipynb`) is included to drive the end-to-end dataset downloading, schema mapping, and training.

---

## Setup & Prerequisites

### Local Installation
```bash
# Install system dependencies
# macOS
brew install espeak-ng libsndfile

# Ubuntu / Debian
sudo apt-get install -y espeak-ng libsndfile1

# Clone the repository along with the custom submodules
git clone https://github.com/sivarajramasamy/koroku-tamil.git
cd koroku-tamil
git clone https://github.com/semidark/StyleTTS2.git StyleTTS2
git clone https://github.com/semidark/kokoro.git kokoro

# Install python requirements
pip install soundfile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
pip install duckdb pyarrow misaki[en]
```

### Environment Variables
Set `BOL_REPO` to point to the repository root directory:
```bash
export BOL_REPO="$(pwd)"
```

---

## Dataset & Training Configs

### 1. Training Config (`configs/config_tamil_ft.yml`)
Points directly to the local parquet files using SQL queries. The dataset is dynamically split 95/5 into training and validation sets:
```yaml
data_params:
  train_data: "SELECT text, gender as speaker_id, row_number() over() - 1 as idx FROM read_parquet('/content/local_tamil_parquet/*.parquet') WHERE text IS NOT NULL AND idx % 20 != 0"
  val_data: "SELECT text, gender as speaker_id, row_number() over() - 1 as idx FROM read_parquet('/content/local_tamil_parquet/*.parquet') WHERE text IS NOT NULL AND idx % 20 == 0"
  root_path: "/content/local_tamil_parquet/*.parquet"
  OOD_data: "../training/OOD_texts.txt"
```

### 2. Model Config (`configs/config_ta.json`)
Specifies the token vocabulary dictionary and architecture settings for the Tamil model.

---

## 📓 Google Colab Training

To train the model on Google Colab:
1. Upload [koroku_tamil_training.ipynb](koroku_tamil_training.ipynb) to Google Colab.
2. Ensure you have accepted the dataset conditions at [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) on Hugging Face.
3. Obtain your Hugging Face Access Token to authorize gated downloads.
4. Select a T4, A100, or L4 GPU runtime and run all cells.

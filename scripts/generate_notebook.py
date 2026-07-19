#!/usr/bin/env python3
import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# koroku-tamil: Training Kokoro-82M on Tamil Rasa Dataset\n",
    "\n",
    "This notebook downloads the gated `ai4bharat/Rasa` dataset from Hugging Face, copies the parquet files locally, configures the SQL-based loading pipeline, and starts the fine-tuning training process using **StyleTTS2**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Install System Dependencies & Clone Repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Verify GPU Allocation\n",
    "!nvidia-smi\n",
    "\n",
    "# 2. Install essential system audio/G2P/libsndfile packages\n",
    "!sudo apt-get update && sudo apt-get install -y espeak-ng libsndfile1\n",
    "\n",
    "# 3. Clone repository along with training submodules\n",
    "!git clone https://github.com/sivarajramasamy/koroku-tamil.git\n",
    "%cd koroku-tamil\n",
    "!git submodule update --init --recursive\n",
    "\n",
    "# 4. Install training framework requirements\n",
    "!pip install soundfile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git -q\n",
    "!pip install duckdb pyarrow \"misaki[en]<0.9.0\" -q\n",
    "\n",
    "print(\"Environment setup complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Login to Hugging Face & Download Dataset\n",
    "\n",
    "Run this cell and enter your HF token (must have accepted dataset conditions for `ai4bharat/Rasa` on HF web interface first)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from huggingface_hub import notebook_login\n",
    "notebook_login()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "import os\n",
    "from huggingface_hub import snapshot_download\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "DRIVE_TARGET_DIR = \"/content/drive/MyDrive/Rasa_Tamil_Dataset/\"\n",
    "os.makedirs(DRIVE_TARGET_DIR, exist_ok=True)\n",
    "\n",
    "print(\"Downloading gated Tamil parquet files from Hugging Face...\")\n",
    "try:\n",
    "    snapshot_download(\n",
    "        repo_id=\"ai4bharat/Rasa\",\n",
    "        repo_type=\"dataset\",\n",
    "        allow_patterns=\"Tamil/*.parquet\",\n",
    "        local_dir=DRIVE_TARGET_DIR,\n",
    "        max_workers=4\n",
    "    )\n",
    "    print(\"🎉 Success! Files written to Google Drive.\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ Download Failed: {e}\")\n",
    "    print(\"Ensure you have accepted terms at https://huggingface.co/datasets/ai4bharat/Rasa on the web.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Copy Parquet files to high-speed local storage\n",
    "\n",
    "FUSE overhead on `/content/drive` is too high for dataloaders during training. We copy the parquets to the local disk first."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import time\n",
    "import shutil\n",
    "import duckdb\n",
    "\n",
    "DRIVE_SRC = \"/content/drive/MyDrive/Rasa_Tamil_Dataset/Tamil/\"\n",
    "LOCAL_DATA_DIR = \"/content/local_tamil_parquet/\"\n",
    "os.makedirs(LOCAL_DATA_DIR, exist_ok=True)\n",
    "\n",
    "# Set MAX_FILES to copy a subset (e.g. 20) for faster start, or None to copy everything\n",
    "MAX_FILES = 20\n",
    "\n",
    "print(f\"Piping Tamil Parquet data (max={MAX_FILES}) onto fast local scratch storage...\")\n",
    "start_time = time.time()\n",
    "\n",
    "parquet_files = sorted([f for f in os.listdir(DRIVE_SRC) if f.endswith(\".parquet\")])\n",
    "files_to_copy = parquet_files[:MAX_FILES] if MAX_FILES is not None else parquet_files\n",
    "\n",
    "for f in files_to_copy:\n",
    "    shutil.copy2(os.path.join(DRIVE_SRC, f), os.path.join(LOCAL_DATA_DIR, f))\n",
    "\n",
    "print(f\"High-speed local copy finished in {time.time() - start_time:.2f} seconds.\")\n",
    "print(f\"Local storage contains: {len(os.listdir(LOCAL_DATA_DIR))} shards.\")\n",
    "\n",
    "# Combine all files into a single optimized parquet for 10x faster dataloader queries\n",
    "print(\"Combining local shards into a single optimized /content/combined_tamil.parquet file...\")\n",
    "con = duckdb.connect()\n",
    "con.execute(\"COPY (SELECT * FROM read_parquet('/content/local_tamil_parquet/*.parquet')) TO '/content/combined_tamil.parquet' (FORMAT PARQUET)\")\n",
    "con.close()\n",
    "print(\"🎉 Parquet optimization complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Verify Schema & Generate OOD Texts\n",
    "\n",
    "This step queries 500 texts, translates them to IPA using espeak-ng/Tamil, and generates `OOD_texts.txt`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import duckdb\n",
    "from misaki.espeak import EspeakG2P\n",
    "\n",
    "con = duckdb.connect()\n",
    "print(\"Parquet Schema Description:\")\n",
    "print(con.execute(\"DESCRIBE SELECT * FROM read_parquet('/content/combined_tamil.parquet') LIMIT 1\").df())\n",
    "\n",
    "print(\"\\nGenerating Tamil Out-Of-Domain texts for Stage 2...\")\n",
    "rows = con.execute(\"SELECT text FROM read_parquet('/content/combined_tamil.parquet') WHERE text IS NOT NULL LIMIT 500\").fetchall()\n",
    "g2p = EspeakG2P(language=\"ta\")\n",
    "\n",
    "ood_lines = []\n",
    "for row in rows:\n",
    "    text = row[0]\n",
    "    try:\n",
    "        ipa, _ = g2p(text)\n",
    "        if ipa.strip():\n",
    "            ood_lines.append(ipa.strip())\n",
    "    except Exception:\n",
    "        pass\n",
    "\n",
    "os.makedirs(\"/content/koroku-tamil/training\", exist_ok=True)\n",
    "with open(\"/content/koroku-tamil/training/OOD_texts.txt\", \"w\", encoding=\"utf-8\") as f:\n",
    "    f.write(\"\\n\".join(ood_lines) + \"\\n\")\n",
    "print(f\"🎉 Generated training/OOD_texts.txt with {len(ood_lines)} lines.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Fetch Pretrained Base Weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd /content/koroku-tamil\n",
    "!mkdir -p /content/koroku-tamil/training\n",
    "!wget -O /content/koroku-tamil/training/kokoro_base.pth https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth\n",
    "\n",
    "# Convert Kokoro-82M base weights to StyleTTS2 format\n",
    "import os\n",
    "os.environ[\"BOL_REPO\"] = \"/content/koroku-tamil\"\n",
    "os.environ[\"BOL_KOKORO_BASE\"] = \"/content/koroku-tamil/training/kokoro_base.pth\"\n",
    "os.environ[\"BOL_KOKORO_CONFIG\"] = \"/content/koroku-tamil/configs/config_ta.json\"\n",
    "!python3 scripts/convert_kokoro_weights.py --force"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Launch Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize accelerate config\n",
    "!accelerate config default\n",
    "\n",
    "# Copy symbols file into StyleTTS2 folder\n",
    "!cp training/kokoro_symbols.py StyleTTS2/kokoro_symbols.py\n",
    "\n",
    "# Set project directory env\n",
    "os.environ[\"BOL_REPO\"] = \"/content/koroku-tamil\"\n",
    "\n",
    "# Start Stage 1 training using the Tamil config\n",
    "!STAGE=1 CONFIG=../configs/config_tamil_ft.yml ./scripts/launch_training.sh"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

target_path = Path(__file__).resolve().parents[1] / "koroku_tamil_training.ipynb"
target_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Generated Jupyter Notebook at {target_path}")

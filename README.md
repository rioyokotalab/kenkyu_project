# 研究プロジェクト（横田研究室）2026年度 秋期

学部3年生が、横田研究室の研究活動を**実際に手を動かして**体験する演習です。
研究室の中心テーマである**大規模言語モデル（LLM）／基盤モデル**を題材に、
TSUBAME4 スーパーコンピュータ上で、**小さな GPT を一から実装して学習・改善**します。
生成AIは「学ぶ対象」であると同時に「研究の道具」でもあります。両方を使いこなします。

## この演習で身につくこと

- 言語モデルの原理を、NumPy の 1 行から GPT まで**自分で組み立てて**理解する
- スーパーコンピュータ（TSUBAME4）での学習ジョブ投入・分散学習・実験管理という**研究のエンジニアリング**
- 生成AI（Claude/Copilot 等）を研究の道具として使い、その出力を**一次情報で検証する**（研究室の文化＝エビデンスファースト）

## 日程（目安・初回確定）

| 回 | 内容 |
| --- | --- |
| 第1回 **10/30(金) 10:30–12:00 石川台9号館 103号室** | オリエン、TSUBAME 環境構築、`git clone` |
| 第2回 | 深層学習の基礎を手で（00–02）、データとトークナイザ（03） |
| 第3回 | 言語モデルを一から（04 bigram → 05 attention → 06 block → 07 GPT） |
| 第4回 | GPU・設定・評価・実験管理（08–11） |
| 第5回 | BPE と日本語（12）、高速化 bf16/Flash/torch.compile（13）、分散 DDP（14） |
| 第6回 | 課題の sweep 結果を発表、研究室の研究（効率化・強化学習・Swallow）紹介 |

※ 第2回以降の日時は第1回で相談して決めます。

## セットアップ（TSUBAME4）

1. Tokyo Tech portal → TSUBAME4 portal でアカウント作成、SSH 鍵を登録（`ssh-keygen`）
2. ログイン後、`~/.bash_profile` に以下を追記して `source ~/.bash_profile`:
   ```bash
   module load nvhpc nccl
   export PATH=/gs/bs/tga-hpc-lecture/$USER/.local/bin:$PATH
   export PYTHONUSERBASE=/gs/bs/tga-hpc-lecture/$USER/.local
   export UV_CACHE_DIR=/gs/bs/tga-hpc-lecture/$USER/.cache/uv
   export UV_VENV_ROOT=/gs/bs/tga-hpc-lecture/$USER/venvs
   ```
3. 環境を作る:
   ```bash
   uv venv $UV_VENV_ROOT/pytorch -p 3.10
   activate pytorch
   git clone https://github.com/rioyokotalab/kenkyu_project.git
   cd kenkyu_project
   uv pip install -e .
   ```
4. データ（`input.txt`, Tiny Shakespeare）は 03/04 の初回実行時に自動ダウンロードされます。

**保存場所の注意**: ホーム（`/home`）は容量が小さい。データやチェックポイントは
`/gs/bs/tga-hpc-lecture/$USER/` 以下に置くこと。

## スクリプトの進め方（00 → 14）

| ファイル | テーマ |
| --- | --- |
| `00_numpy.py` | NumPy だけで forward/backward を手書き |
| `01_tensors.py` | PyTorch テンソル |
| `02_autograd.py` | 自動微分 |
| `03_data.py` | テキスト・文字トークナイザ・バッチ（共有コードは `data_utils.py`） |
| `04_bigram.py` | 最小の言語モデル（bigram） |
| `05_attention.py` | self-attention を一から |
| `06_block.py` | マルチヘッド注意＋FFN＝Transformer ブロック |
| `07_gpt.py` | 小さな GPT を学習・生成（モデル本体は `gpt.py`） |
| `08_gpu.py` | GPU で tokens/sec を測る |
| `09_config.py` | 設定（dataclass）とチェックポイント再開 |
| `10_eval.py` | perplexity と進捗表示 |
| `11_wandb.py` | Weights & Biases で実験記録 |
| `12_bpe.py` | BPE（tiktoken）と日本語 |
| `13_amp.py` | bf16・torch.compile・Flash Attention・KV-cache・勾配累積 |
| `14_ddp.py` | 分散データ並列（torchrun）。課題と sweep の本体 |

### 動かし方

```bash
# ログインノードで軽く動作確認（CPU でも可、小さく）
python 03_data.py
python 04_bigram.py

# 計算ノードで GPU 実行（対話）
qrsh -g tga-hpc-lecture -l gpu_1=1 -l h_rt=1:00:00
activate pytorch
python 07_gpt.py

# 1 ノード 4 GPU の分散学習（バッチジョブ）
qsub -g tga-hpc-lecture job_ddp.sh
```

## 課題（提出締切は授業で案内）

`14_ddp.py` をベースに、`sweep.yaml` でハイパーパラメータを探索し、
**検証 perplexity をできるだけ下げる**こと。

```bash
wandb login
wandb sweep sweep.yaml                 # 表示される sweep ID を控える
qsub -g tga-hpc-lecture job.sh <rioyokotalab/kenkyu_project/SWEEP_ID>
```

提出物（wandb のレポート 1 枚にまとめる）:
1. 学習曲線（train/val loss、val perplexity）
2. 見つけた最良設定とその val perplexity
3. その設定での生成テキスト例
4. うまくいかなかった設定と、その原因の考察（←研究で最重要）

発展（任意）: bf16（13）、Flash Attention、文脈長を伸ばす、BPE（12）で日本語を学習。

### 採点のめやす
- 動く sweep と結果の記録（40%）
- 考察の質（なぜその結果か、何を変えると何が起きたか）（40%）
- 発展的な工夫・生成AIの使い方と検証（20%）

## 生成AIの使い方

本演習では Claude / Copilot 等の利用を**歓迎**します。ただし研究室のルールとして、
**AI の出力は必ず自分で読んで理解し、一次情報（公式ドキュメント・実際の挙動）で検証**してください。
詳しくは [`ai_assisted_coding.md`](ai_assisted_coding.md)。レポートには AI を使った箇所を明記すること。

## 困ったとき（FAQ）

- **ジョブが流れない / 待たされる**: 資源を多く使うジョブは待ち時間が長い。まず `gpu_1=1` の小さなジョブで確認。
- **`CUDA out of memory`**: `batch_size` か `block_size` を下げる、または `--grad_accum` を増やす。
- **wandb がネットに繋がらない**: 計算ノードは外部不可のことがある。`export WANDB_MODE=offline` で実行し、ログインノードで `wandb sync`。
- **`loss` が nan**: 学習率が高すぎる。`--lr` を下げる。
- **生成が支離滅裂**: 学習不足（`--max_iters` を増やす）か、モデルが小さすぎる。

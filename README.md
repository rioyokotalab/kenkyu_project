# 研究プロジェクト（横田研究室）2026年度 秋期

学部3年生が、横田研究室の研究活動を**実際に手を動かして**体験する演習です。研究室の中心テーマである
**大規模言語モデル（LLM）**を題材に、TSUBAME4 上で**小さな GPT を一から実装し、学習・改善**します。

- 言語モデルの原理を NumPy の 1 行から GPT まで自分で組み立てて理解する
- スパコン（TSUBAME4）での学習ジョブ投入・分散学習・実験管理という研究のエンジニアリングを身につける
- 生成AI（Claude/Copilot）を研究の道具として使い、出力を一次情報で検証する（エビデンスファースト）

## 日程

| 回 | 内容 |
| --- | --- |
| 第1回 **10/30(金) 10:30–12:00 石川台9号館 103** | オリエン、TSUBAME 環境構築、`git clone`、00–03 |
| 第2回（第1回で決定） | 言語モデルを一から：04 bigram → 05 attention → 06 block → 07 GPT |
| 第3回 | GPU・引数・評価・wandb：08 → 11 |
| 第4回 | BPE と日本語、高速化（bf16/Flash/KV-cache）、分散 DDP：12 → 14 |
| 第5回 | 課題（sweep）の発表、研究室の研究紹介 |

## セットアップ（TSUBAME4）

1. Tokyo Tech portal → TSUBAME4 portal でアカウント作成、SSH 鍵（`ssh-keygen -t ed25519`）を登録
2. `~/.bash_profile` に追記して `source ~/.bash_profile`:
   ```bash
   export PATH=/gs/bs/tga-hpc-lecture/$USER/.local/bin:$PATH
   export PYTHONUSERBASE=/gs/bs/tga-hpc-lecture/$USER/.local
   export UV_CACHE_DIR=/gs/bs/tga-hpc-lecture/$USER/.cache/uv
   export UV_VENV_ROOT=/gs/bs/tga-hpc-lecture/$USER/venvs
   export UV_CONCURRENT_INSTALLS=4    # ログインノードはプロセス数上限が小さい
   activate() { source $UV_VENV_ROOT/$1/bin/activate; }
   ```
3. 環境を作る（一度だけ）:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh    # uv を入れる
   source ~/.bash_profile
   uv venv $UV_VENV_ROOT/pytorch -p 3.10
   activate pytorch
   cd /gs/bs/tga-hpc-lecture/$USER
   git clone https://github.com/rioyokotalab/kenkyu_project.git
   cd kenkyu_project
   uv pip install -e .
   python 03_data.py          # 動作確認（input.txt を自動ダウンロード）
   ```

**注意**: `/home` は容量が小さい。venv・データ・チェックポイントは `/gs/bs/tga-hpc-lecture/$USER/` に置く。

## スクリプト（00 → 14、1 つずつ機能が増える）

07 以降は**前のスクリプトに 1 機能だけ足した差分**になっている。`diff 07_gpt.py 08_gpu.py` のように
見比べると、何を足したかが分かる。

| ファイル | 足したもの |
| --- | --- |
| `00_numpy.py` | NumPy だけで forward / backward / update を手書き |
| `01_tensors.py` | PyTorch テンソル（NumPy との対応） |
| `02_autograd.py` | 自動微分：`backward()` が 00 の逆伝播を全部やる |
| `03_data.py` | テキスト → 整数列、文字トークナイザ、(x, y) バッチ（共有コードは `data_utils.py`） |
| `04_bigram.py` | 最小の言語モデル（埋め込み 1 枚）。学習ループと生成の型 |
| `05_attention.py` | self-attention を一から（Q, K, V、√d、因果マスク） |
| `06_block.py` | マルチヘッド注意＋FFN＋残差＋LayerNorm = Transformer ブロック（`gpt.py` の Block） |
| `07_gpt.py` | 小さな GPT を学習・生成（モデルは `gpt.py`）。**以降の基準スクリプト** |
| `08_gpu.py` | 07 + GPU、tokens/sec の計測 |
| `09_args.py` | 08 + argparse（ハイパーパラメータを引数に）、seed |
| `10_meter.py` | 09 + perplexity、AverageMeter |
| `11_wandb.py` | 10 + Weights & Biases（学習曲線・生成例の記録） |
| `12_bpe.py` | 11 + BPE トークナイザ（tiktoken）。`--tokenizer bpe --data japanese.txt` で日本語も |
| `13_amp.py` | 12 + bf16・Flash Attention・torch.compile・勾配累積・KV-cache |
| `14_ddp.py` | 13 + 分散データ並列（torchrun、`node_f=1` で 4 GPU）。**課題と sweep の本体** |

### 動かし方

```bash
# ログインノードで軽い確認（CPU）
python 03_data.py && python 04_bigram.py

# 計算ノードで GPU（対話）
qrsh -g tga-hpc-lecture -l gpu_1=1 -l h_rt=1:00:00
activate pytorch
python 08_gpu.py
python 09_args.py --n_layer 6 --n_embd 384 --block_size 256 --max_iters 5000

# 1 ノード 4 GPU（バッチ、node_f=1 = H100 ×4。node_q=1 は 1/4 ノード = GPU 1 枚）
mkdir -p output
qsub -g tga-hpc-lecture job_ddp.sh
```

## 課題

`14_ddp.py` を `sweep.yaml` で探索し、**検証 perplexity（val_ppl）をできるだけ下げる**。

```bash
activate pytorch
wandb login                              # 初回のみ（https://wandb.ai/authorize のキーを貼る）
wandb sweep sweep.yaml                   # 表示される「wandb agent <自分のID>/kenkyu_project/<SWEEP_ID>」を控える
mkdir -p output
qsub -g tga-hpc-lecture job.sh <自分のID>/kenkyu_project/<SWEEP_ID>   # 複数投げると並列に探索
```

提出物（wandb のレポート 1 枚、締切は授業で案内）:
1. 学習曲線（train/val loss、val_ppl）
2. 最良設定とその val_ppl
3. その設定での生成テキスト例
4. うまくいかなかった設定と、その原因の考察（研究で最も重要）
5. 生成AIを使った箇所と、どう検証したか

採点のめやす: 動く sweep と記録 40% ／ 考察の質 40% ／ 工夫・AI の検証 20%。
発展（任意）: bf16/Flash で tokens/sec を上げる、文脈長を伸ばす、BPE で日本語学習、top-p サンプリング実装。

## 生成AIの使い方

Claude / Copilot 等の利用は歓迎します。ただし **AI の出力は必ず自分で読んで理解し、一次情報と実際の
挙動で検証**してください（[`ai_assisted_coding.md`](ai_assisted_coding.md)）。レポートに AI を使った箇所を明記すること。

## TSUBAME4 の資源指定（実機確認 2026-10-18）

| 指定 | 内容 |
| --- | --- |
| `-l gpu_1=1` | GPU 1 枚（演習の基本。sweep の 1 試行もこれ） |
| `-l node_q=1` | 1/4 ノード = GPU 1 枚（`gpu_1` と同じ枚数。CPU/メモリが多い） |
| `-l node_h=1` | 1/2 ノード = GPU 2 枚 |
| `-l node_f=1` | ノード 1 台 = H100 4 枚（`job_ddp.sh`） |

H100 1 枚での実測（2026-10-18）: 08_gpu（0.8M params, block 64）2000 step 21.3 秒 ≈ 192,000 tokens/sec（実行ごとに 1〜2 割変動）、
GPU メモリ最大 0.17 GB。13_amp（10.7M params, block 256, bf16+Flash）≈ 176,000 tokens/sec。
`--compile` は 500 step の短い実行ではコンパイル時間が勝って遅くなる（長い学習で効く）。

## 困ったとき

- **ジョブが流れない**: 資源を多く使うほど待つ。まず `gpu_1=1`・1 時間で確認。`qstat` で状態を見る
- **CUDA out of memory**: `--batch_size` / `--block_size` を下げる、`--grad_accum` を増やす
- **wandb が繋がらない**: `export WANDB_MODE=offline` で実行し、ログインノードで `wandb sync`
- **loss が nan**: `--lr` を下げる（1e-4 程度）
- **生成が支離滅裂**: 学習不足（`--max_iters` を増やす）かモデルが小さい
- **ModuleNotFoundError**: `activate pytorch` を忘れている／`uv pip install -e .` 未実行

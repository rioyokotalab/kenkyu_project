"""07_gpt.py — 小さな GPT を学習して文章を生成する（以降の 08〜14 はこのファイルに 1 機能ずつ足していく）。

学ぶこと:
- gpt.py の GPT（埋め込み + Transformer ブロック×n_layer + 出力層）を学習する。
- 04 の bigram と比べ、文脈を見る分だけ生成がそれらしくなることを体感する。

試すこと: n_layer / n_embd / block_size を変えて loss と生成文がどう変わるか。
"""
import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

device = "cpu"                      # 08 で GPU にする
torch.manual_seed(1337)
max_iters, eval_interval, eval_iters = 2000, 200, 50
batch_size, lr = 32, 3e-4
cfg = GPTConfig(block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.1)


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device)
        _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    train_data, val_data, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    print(f"params={model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            print(f"step {it:5d}  train {tr:.4f}  val {va:.4f}")
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tok.decode(model.generate(start, 500, temperature=1.0, top_k=50)[0].tolist()))


if __name__ == "__main__":
    main()

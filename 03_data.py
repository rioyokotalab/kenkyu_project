"""03_data.py — テキストデータと文字トークナイザ（char-level）を理解する。

学ぶこと:
- 言語モデルは「次のトークンを当てる」問題。まずトークン＝文字（char）として扱う。
- テキストを整数列に符号化（encode）・復号（decode）する。
- block_size（文脈長）で切り出したミニバッチ (x, y) を作る。y は x を 1 つずらしたもの。

試すこと:
- block_size / batch_size を変えて x, y の形がどう変わるか確認する。
- input.txt を自分の好きなテキストに差し替えてみる。

共有コードは data_utils.py にある（04 以降はそこから import する）。
"""
from data_utils import get_batch, load_dataset


def main():
    train_data, val_data, tok = load_dataset()
    print(f"語彙数 (vocab_size): {tok.vocab_size}")
    print("符号化の例:", tok.encode("Hello"))
    print("復号の例:", tok.decode(tok.encode("Hello")))
    print(f"train: {len(train_data):,} tokens / val: {len(val_data):,} tokens")

    x, y = get_batch(train_data, block_size=8, batch_size=4)
    print("x の形:", tuple(x.shape), " y の形:", tuple(y.shape))
    print("x[0]:", x[0].tolist())
    print("y[0]:", y[0].tolist(), "  <- x を 1 つ左にずらした「次の文字」")


if __name__ == "__main__":
    main()

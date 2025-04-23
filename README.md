# optimal-auctions
```text
Deep Learning による最適オークション — 非公式実装
論文: Optimal Auctions through Deep Learning (ICML 2019) — Paul Dütting, Zhe Feng, Harikrishna Narasimham, David C. Parkes, Sai S. Ravindranath
リポジトリ状態: アルファ版（研究用途向け／本番運用は想定していません）
🙏 論文著者への感謝
本リポジトリは上記論文のアイデアを参考に 一から実装 したものです。素晴らしい研究成果を公開してくださった著者の皆様に深く感謝いたします。
```

# 機能

RegretNet 実装

論文で提案されたニューラルネットワーク・アーキテクチャを PyTorch で再構築。Additive / Unit‑Demand / Combinatorial 3 種のバリュエーションをサポート。

柔軟なトレーニングパイプライン

学習データの分布・ネットワークサイズ・損失関数パラメータ (ρ, λ 等) を YAML で定義。

再現可能な実験

実験用ノートブックとスクリプト (scripts/ 配下) を用意。論文の主要図表をワンコマンドで再生成可能。

GPU & マルチラン対応

PyTorch Lightning / Hydra を利用し、複数 GPU やハイパーパラメータ掃きも簡単。

# リポジトリ構成
```text
optimal-auctions/
├── README.md              # ← 今ご覧のファイル
├── LICENSE
├── pyproject.toml         # Poetry 環境定義（pip を使う場合は requirements.txt も用意）
├── .gitignore
│
├── src/
│   ├── auctions/          # コアロジック
│   │   ├── networks.py    # RegretNet モデル定義
│   │   ├── loss.py        # 収益＋後悔(レグレット)計算
│   │   ├── trainer.py     # 学習ループ (Augmented Lagrangian 法)
│   │   └── ...
│   └── utils/             # 補助関数・データ生成など
│
├── scripts/               # 実験用コマンド
│   ├── train.py           # 実装のエントリポイント
│   └── eval.py            # 評価・可視化
│
├── notebooks/             # 解析・図表生成 Jupyter NB
└── tests/                 # 単体テスト (pytest)
```

# セットアップ手順

Poetry を使う場合（推奨）

## 1. リポジトリをクローン
```text
$ git clone https://github.com/yourname/optimal-auctions.git
$ cd optimal-auctions
```

## 2. 仮想環境の作成＆依存関係インストール
```text
$ poetry install
```

## 3. シェルに入る

```text
$ poetry shell

pip のみで試す場合

$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

## 使い方

トレーニング

python scripts/train.py \
    --config configs/additive2x2.yaml \
    --gpus 1

主な引数:

--config: データ分布・ネットワーク規模・学習率などを記述した YAML

--gpus: 使用 GPU 数（CPU の場合は 0）

評価 & 可視化
```text
python scripts/eval.py --checkpoint path/to/ckpt.ckpt
```

Jupyter Notebook

実験結果の再現やグラフ描画は notebooks/ 内の NB を起動して行えます。

## モデル概要
```text
RegretNet は 収益最大化 と dominant‑strategy IC を同時に考慮するニューラルネットワークベースのオークション設計フレームワークです。

入力: 入札者 × アイテムの価値ベクトル

出力: 各入札者への割当確率行列 g(b) と支払い p(b)

学習目的:

期待収益を最大化 (負の収益を最小化)

制約: 期待 ex‑post regret = 0 (DSIC 達成)

最適化: Augmented Lagrangian 法により、制約付き最適化を勾配ベースで解く

詳細数式は論文 §2–4 を参照してください。
```

## 実験の再現

```text
論文 Table 1–5 の設定を configs/ に用意しています。
for cfg in configs/icml19/*.yaml; do
  python scripts/train.py --config $cfg --gpus 1
done
完了後、results/ に CSV ログとチェックポイントが保存されます。
```

## 引用

研究に本実装を用いた場合は、論文原著を必ず引用してください。
```text
@inproceedings{duetting2019optimal,
  title={Optimal Auctions through Deep Learning},
  author={D{"u}tting, Paul and Feng, Zhe and Narasimham, Harikrishna and Parkes, David C. and Ravindranath, Sai S.},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}
}
```

## ライセンス

本リポジトリのコードは MIT License で配布します。詳細は LICENSE を参照してください。

## コントリビュート方法

Issue を立てて議論

feature/your-topic ブランチで開発

pytest をパスさせる

Pull Request を送る

バグ報告や機能提案なども歓迎です！

# XRD Analyzer

X線回折（XRD）パターン解析のためのStreamlitアプリケーションです。NMF（非負値行列因子分解）とDBSCANクラスタリングを用いて、XRDパターンの自動分類と可視化を行います。

## Features

- **データ読み込み**: 複数のXRDファイルフォーマットに対応
  - `.xy`, `.txt`: 2カラム形式（2θ, 強度）
  - `.csv`: CSVフォーマット
  - `.ras`: Rigaku RASフォーマット

- **前処理機能**:
  - 強度の正規化
  - 2θ範囲のトリミング
  - バックグラウンド除去（BEADSアルゴリズム）
  - スムージング（Savitzky-Golay フィルター）
  - データ補間

- **NMF解析**:
  - XRDパターンを基底パターン（成分）に分解
  - 最適な成分数の自動推定
  - 再構成誤差の可視化

- **クラスタリング**:
  - DTW（Dynamic Time Warping）距離計算
  - MDS/t-SNE/UMAPによる次元削減
  - DBSCANクラスタリング

- **可視化**:
  - インタラクティブなXRDパターン表示
  - ヒートマップ表示
  - クラスタ分布図
  - 三元相図（Ternary Plot）
  - 確率分布マップ

## Installation

### 1. 環境のセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/matsui-naoki/xrd-app.git
cd xrd-app/xrd-app

# 仮想環境の作成（オプション）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 追加の依存パッケージ（オプション）

```bash
# バックグラウンド除去機能を使用する場合
pip install pybeads

# DTW距離計算を使用する場合
pip install dtw-python

# UMAP次元削減を使用する場合
pip install umap-learn
```

## Usage

### アプリケーションの起動

```bash
streamlit run app.py
```

または

```bash
./run.sh
```

### 基本的な使い方

1. **データのアップロード**
   - サイドバーの「Data Upload」セクションでXRDファイルをアップロード
   - 複数ファイルの一括アップロードに対応

2. **前処理**
   - 「Preprocessing」セクションで前処理オプションを設定
   - 「Apply Preprocessing」ボタンをクリック

3. **解析**
   - 「Analysis」セクションでNMFとクラスタリングのパラメータを設定
   - 「Run Analysis」ボタンをクリック

4. **結果の確認**
   - メインパネルの各タブで結果を確認
   - 「Results」タブからCSVファイルとしてエクスポート可能

## Project Structure

```
xrd-app/
├── app.py                 # メインStreamlitアプリケーション
├── requirements.txt       # 依存パッケージ
├── README.md             # このファイル
├── run.sh                # 起動スクリプト
├── .gitignore            # Git除外設定
│
├── tools/                # 解析ツール
│   ├── __init__.py
│   ├── data_loader.py    # ファイル読み込み
│   ├── preprocessing.py  # 前処理関数
│   └── analysis.py       # NMF, クラスタリング
│
├── components/           # UIコンポーネント
│   ├── __init__.py
│   ├── plots.py          # Plotly可視化関数
│   └── styles/           # CSSスタイル
│       ├── __init__.py
│       └── custom.css
│
└── utils/                # ユーティリティ
    ├── __init__.py
    ├── helpers.py        # ヘルパー関数
    └── help_texts.py     # ヘルプテキスト
```

## Parameters Guide

### 前処理パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| Normalize | 最大強度を100に正規化 | ON |
| 2θ Range | 解析する2θ範囲 | 10-60° |
| Remove Background | BEADSによるバックグラウンド除去 | ON |
| Smoothing | Savitzky-Golayスムージング | OFF |

### NMFパラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| Number of Components | 基底パターンの数 | 10 |

### クラスタリングパラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| Distance Method | 距離計算方法（DTW/cosine/correlation） | DTW |
| DTW Window Size | DTWのウィンドウサイズ | 30 |
| Dimension Reduction | 次元削減方法（MDS/t-SNE/UMAP） | MDS |
| DBSCAN eps | クラスタ半径 | 20 |
| DBSCAN min_samples | 最小サンプル数 | 1 |

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Material-Explorer project for the analysis algorithms
- EIS-app for the UI design reference

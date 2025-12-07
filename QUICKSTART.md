# XRD Analyzer クイックスタートガイド

5分でXRD Analyzerを始めましょう。

## 1. インストール

```bash
# リポジトリのクローン
git clone https://github.com/matsui-naoki/xrd-app.git
cd xrd-app

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 2. アプリの起動

```bash
streamlit run app.py
```

ブラウザが自動的に開きます（通常は http://localhost:8501）

## 3. データのアップロード

1. 左サイドバーの「Data Upload」セクションを開く
2. 「Browse files」をクリックしてXRDファイルを選択
3. 複数ファイルを一度にアップロード可能

**対応フォーマット:** `.xy`, `.txt`, `.csv`, `.ras`, `.dat`

## 4. 前処理の適用

1. 「Preprocessing」セクションを開く
2. 必要なオプションを選択：
   - Normalize: 強度を正規化
   - 2θ Range: 解析範囲を指定
   - Remove Background: バックグラウンド除去
3. 「Apply Preprocessing」をクリック

## 5. 解析の実行

1. 「Analysis」セクションを開く
2. パラメータを設定：
   - Number of Components: NMFの成分数（推奨: 5-15）
   - Distance Method: DTW（推奨）
   - DBSCAN eps: クラスタサイズ（推奨: 10-30）
3. 「Run Analysis」をクリック

## 6. 結果の確認

メインパネルの各タブで結果を確認：

- **Data**: XRDパターンの表示
- **Preprocessing**: 前処理の効果
- **Analysis**: NMFとクラスタリング結果
- **Mapping**: 三元相図
- **Results**: 結果テーブルとエクスポート

## 7. 結果のエクスポート

「Results」タブで：
- 「Download Results CSV」: 分類結果
- 「Download Basis Vectors」: NMF基底パターン
- 「Download Coefficients」: NMF係数

## ヒント

- 最適なNMF成分数は「Find Optimal Components」ボタンで探索
- クラスタ数が多すぎる場合はDBSCAN epsを大きくする
- セッションは「Session Management」で保存/復元可能

## トラブルシューティング

**Q: ファイルが読み込めない**
A: ファイルが2カラム形式（2θ, 強度）であることを確認

**Q: 解析が遅い**
A: サンプル数が多い場合、cosine距離を試す（DTWより高速）

**Q: クラスタが1つしかない**
A: DBSCAN epsを小さくする

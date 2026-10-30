# 再現できる範囲と実行手順

## 保存済み出力からの再計算

READMEの手順で `scripts/reproduce.py` を実行します。採点は `grain_prices.canonical.relations.evaluate_relations` を呼び出します。位置の一致、値・単位・貨幣・単価の根拠までの完全一致、対象単位のペア有無、欠落・検査失敗を照合します。HTTPやAPIへの接続は再計算スクリプト内で禁止しています。

`provenance/code_at_execution.json` は、実験の固定記録に保存されたコードハッシュと公開コードの対応を示します。`execution_sha256` がないファイルは実験のコード記録の対象外だったことを意味し、同一性を推測しません。プロンプトv2と出力スキーマのハッシュは `provenance/experiment.json` に記載します。

## 候補生成と新しいモデル実行

抽出処理の入口は `grain-prices` です。候補生成にはDDbDP・HGVを別途取得して設定します。コーパス全体はこの配布物に含めません。

```sh
uv run grain-prices --help
uv run grain-prices build-spine --help
uv run grain-prices generate-candidates --help
uv run grain-prices relations --help
```

新しいモデル呼出しは、保存済み結果の再計算とは別の実験です。APIキー、利用可能なモデル、料金と呼出し上限の設定が別途必要です。READMEの再計算・テスト・CIでは有料APIを呼び出しません。

`legacy/stage2.py` 等は、現在の正規化・数値展開・トークン分割から参照される互換コードとして含めています。旧実験の注釈・旧評価値を今回の評価に混ぜていません。

## 検証

WindowsとLinuxで、配布一覧・ハッシュ検査、6条件の再計算、選定した処理テストをCIで実行します。テスト中のLLM応答は模擬データです。集計のみ公開する全件適用について検査できるのは表間・合計の整合であり、全件の原出力からの再集計ではありません。

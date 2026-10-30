# じんもんこん2026 穀物の数量・金額ペア抽出と評価

ローマ期エジプトのパピルス文書から穀物の数量・金額と根拠を抽出する研究の、論文対応コードと評価資料です。**150対象・36ペアを基準とした3モデル×2入力長の評価を、保存済み出力から再計算できます。APIキーや外部コーパスは再計算に不要です。**

論文対応版: **jmc2026-v1.0.0**。本文の表4に対応する評価資料と、表5・6の全件適用の集計を収録します。論文本文はこのリポジトリに含めません。

## 評価を再計算する

Python 3.12と[uv](https://docs.astral.sh/uv/)を使用します。依存パッケージの初回取得にはネットワーク接続が必要ですが、再計算時には接続しません。

```sh
uv sync --locked --all-extras
uv run python scripts/check_release.py
uv run python scripts/reproduce.py
uv run pytest -q --basetemp .tmp -p no:cacheprovider
```

`reproduced/table4.csv` に評価値、`reproduced/` 内のJSONに各条件の採点と対象ごとの不一致が出力されます。再計算には実験で用いた評価器を使い、保存結果との一致を検査します。新しいLLM出力を生成するコマンドではありません。

## 収録内容

| 場所 | 内容 |
|---|---|
| `src/grain_prices/` | 正規化・候補抽出・モデル呼出し・出力検査・評価・重複整理と依存コード |
| `src/grain_prices/prompts/target_grain_relations_v2.md` | 今回の実験に使ったプロンプト。v1などは既存コード・テストの互換性のため保持 |
| `data/evaluation/` | 150対象の参照データ、前後25語・80語の入力、6条件の保存済み出力 |
| `results/` | 表4の期待値と、表5・6に対応する全件適用の集計 |
| `provenance/` | 元ファイルと公開ファイルのハッシュ、出典、出力形式、実行条件とコード版の照合 |
| `docs/` | データ形式、再現できる範囲と手順、利用条件 |

## 結果の範囲

150対象のうち32対象に36ペアがあり、118対象にはペアがありません。抽出規則やプロンプトの検討に用いた資料を含み、独立した評価標本ではありません。人手の参照は一人が確認し、各モデル・入力長での実行は1回です。25語では2ペアの根拠の一部が入力範囲外になりますが、同じ36ペアを基準に評価します。

全件適用の9,222対象から出力された2,080ペアは、人手未確認の候補です。2,073群への整理を含む**集計のみ**を公開し、全件の個別出力は配布しません。この配布物だけでは全件出力から集計を作り直すことはできません。

## 出典と利用条件

コードはMIT、自作注釈・説明文はCC BY 4.0です。埋め込まれたDDbDPの電子校訂本文とHGV由来の情報は元のCC BY 3.0を保持します。資料別出典、正規化の説明、適用範囲は[NOTICE](NOTICE.md)と[データの説明](docs/DATA.md)を参照してください。原物画像・コーパス全体・第三者の価格表は配布していません。

引用情報は[CITATION.cff](CITATION.cff)にあります。訂正は新しい版として記録し、論文対応タグを移動しません。

## English summary

This paper companion provides source code, the v2 prompt, a reviewed reference of 150 grain mentions with 36 quantity–amount pairs, and saved outputs from three models at two context lengths. The original evaluator reproduces all six conditions without provider calls. The sample includes development sources, was reviewed by one person, and each condition was run once. Full-population results are distributed only as aggregates; the 2,080 predicted pairs have not been human-verified. Embedded DDbDP texts and HGV metadata retain their source attribution and CC BY 3.0 terms.

# 利用条件と第三者資料

- `src/`、`tests/`、`scripts/` の自作コード: MIT（LICENSE）。
- 自作の注釈、集計、説明文: CC BY 4.0（LICENSE-DATA.md）。
- 評価データに含む電子校訂本文: Duke Databank of Documentary Papyri / Duke Collaboratory for Classics Computing (DC3), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。
- 入力中の年代・地名: Heidelberger Gesamtverzeichnis der griechischen Papyrusurkunden Ägyptens (HGV), CC BY 3.0。これらは入力の識別情報を維持するため保存されますが、LLMに渡される本文の範囲は `visible_input` により限定されます。

利用条件は[idp.dataの公式説明](https://github.com/papyri/idp.data#license)で確認しました（2026-10-30）。DDbDPは配布に含む142文書のローカルXMLの利用条件表示も検査しています。各文書の出典URL・XMLハッシュ・本文ブロックのハッシュは `provenance/source_documents.json` にあります。注釈へのCC BY 4.0の表示は、埋め込まれた本文やメタデータの利用条件を置き換えません。

本文は電子校訂から略記・数値属性を展開し、半量記号を `1/2` に置換し、空白を整理しています。改行等を表す処理用マーカーも残ります。原物写真や外交的翻刻の再現ではありません。LLMは指定範囲のテキストを入力とし、画像を入力としていません。

コーパス全体、原物画像、HarperやRathboneの価格表、学会テンプレート、原稿・査読資料は含みません。Git履歴は公開資料から新規に作成しています。

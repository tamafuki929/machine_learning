# Tabular Data Analysis

## Tips
### Feature Engineering
##### trainとtestで分布が一致しているかに着目する 
- 各特徴量がtrainかtestかで分布が異なるかを調べる
- ハズレ値や裾の長さが異なる場合や, スパイクやbumpの分布に際がある場合, binningで対処(qcut)
- それ以外の大きな分布のズレがある場合, cv時にそれを意識したデータサンプリング(foldの決定)を行う必要がある. 
##### １変数からの特徴量作成は網羅的に行う
- ある未加工の説明変数に着目した際に, その変数から何らかの特徴量を作成できる場合, とりあえず作成し, その後有効性を検証する
- 連続値や綺麗にカテゴライズされたカラムよりも, 複雑なカラムの方が多くの特徴量の作成が可能である. これは, 複雑＝情報量が多いということであるため. 


## Preprocessing
* sklearnを使用した欠損値補完についてはサイト[^4]を参照  
* pipelineの活用に関してはサイト[^5][^6]を参照

## GBDT 

### LightGBM
* LightGBMとXGBの比較についてはnotebook[^2]を参照
* LightGBMのtuningについてはnotebook[^3]を参照
* LightGBMのcvの仕様は文献[^1]にあるように, 特徴的なものとなっている

[^1]: https://blog.amedama.jp/entry/lightgbm-cv-implementation

[^2]: https://www.kaggle.com/code/bextuychiev/how-to-beat-the-heck-out-of-xgboost-with-lightgbm/notebook

[^3]: https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding

[^4]: https://qiita.com/FukuharaYohei/items/9830d5760595619352a5

[^5]: https://dev.classmethod.jp/articles/concatenate_result_with_featureunion/

[^6]: https://scikit-learn.org/stable/modules/compose.html#feature-union
# Next Basket Recommendation with Time-Aware Item Weighting

add description here

```bash
.
├── nbr // NBR library
│   ├── common // subpackage with recommendation metrics definition and common constants
│   ├── dataset // subpackage with NBR dataset definition
│   ├── model // subpackage with NBR models definition
│   ├── preparation // subpackage with dataset preprocessing classes
│   ├── trainer // subpackage with NBR trainer definition
│   └── __init__.py
└── notebooks // folder with experiments on three real-world datasets
    ├── data // folder with unprocessed datasets
    │   ├── dunnhumby.txt // Dunnhumby dataset
    │   ├── ta_feng.txt // TaFeng dataset
    │   └── taobao.txt // TaoBao dataset (purchase interactions only)
    ├── testing_dnntsp // folder with testing DNNTSP model
    │   ├── DNNTSP // DNNTSP library (author's version with minor modifications)
    │   └── testing_dnntsp.ipynb // notebook with testing DNNTSP model
    ├── testing_baselines.ipynb // notebook with testing TopPopular and TopPersonal models
    ├── testing_repurchasemodule.ipynb // notebook with testing Repurchase Module only
    ├── testing_slrc.ipynb // notebook with testing SLRC model
    ├── testing_taiw.ipynb // notebook with testing TAIW model
    ├── testing_taiwi.ipynb // notebook with testing TAIWI model
    ├── testing_tifuknn.ipynb // notebook with testing TIFUKNN model
    ├── testing_tifuknntd.ipynb // notebook with testing TIFUKNN-TA model
    └── testing_upcf.ipynb // notebook with testing UPCF model
```

This repository modified based on [TAIW](https://github.com/alexeyromanov-hse/time_aware_item_weighting) (Romanov, A., Lashinin, O., Ananyeva, M. and Kolesnikov, S., 2023, September. Time-Aware Item Weighting for the Next Basket Recommendations. In Proceedings of the 17th ACM Conference on Recommender Systems (pp. 985-992).) [[pdf]](https://arxiv.org/pdf/2307.16297.pdf).)

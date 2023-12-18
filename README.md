# Next Basket Recommendation with Time-Aware Item Weighting

This repository modified based on [TAIW](https://github.com/alexeyromanov-hse/time_aware_item_weighting) (Romanov, A., Lashinin, O., Ananyeva, M. and Kolesnikov, S., 2023, September. Time-Aware Item Weighting for the Next Basket Recommendations. In Proceedings of the 17th ACM Conference on Recommender Systems (pp. 985-992).) [[pdf]](https://arxiv.org/pdf/2307.16297.pdf).)

## Project Structure

```
.
├── EMOS // folder with experiments on EMOS dataset
│   ├── best_checkpoint-emos-*.pth // best pre-trained model from each period scenario
│   ├── best_checkpoint.pth // overall best pre-trained model
│   ├── Prep-emos.ipynb // notebook for preprocessing
│   └── TAIWI-emos.ipynb // notebook for doing training and inference
├── EPM // folder with experiments on EPM dataset
│   ├── best_checkpoint-epm-*.pth // best pre-trained model for each period scenario
│   ├── best_checkpoint.pth // overall best pre-trained model
│   ├── Prep-epm.ipynb // notebook for preprocessing
│   └── TAIWI-epm.ipynb // notebook for doing training and inference
├── nbr // NBR library
│   ├── common // subpackage with recommendation metrics definition and common constants
│   ├── dataset // subpackage with NBR dataset definition
│   ├── model // subpackage with NBR models definition
│   ├── preparation // subpackage with dataset preprocessing classes
│   ├── trainer // subpackage with NBR trainer definition
│   └── __init__.py
├── notebooks // folder with experiments on three real-world datasets
│   ├── data // folder with unprocessed datasets
│   ├   ├── dunnhumby.txt // Dunnhumby dataset
│   ├   ├── ta_feng.txt // TaFeng dataset
│   ├   └── taobao.txt // TaoBao dataset (purchase interactions only)
│   ├── testing // folder with testing DNNTSP model
│   ├   ├── DNNTSP // DNNTSP library (authors version with minor modifications)
│   └── └── testing_*.ipynb // notebook for testing various model
├── Tafeng // folder with experiments on Tafeng dataset
│   ├── best_checkpoint.pth // overall best pre-trained model
│   └── TAIWI-Tafeng.ipynb // notebook for doing training and inference
├── .gitignore // ignored files by git
├── README.md
├── requirements.txt // requirements for installing python dependencies
└── script.py // script for doing the training process automatically (equivalent to the notebook)
```

## Preparation

### Installation

```bash
# install requirements
pip install -r requirements.txt
```

### Data

Download [dataset](https://univindonesia-my.sharepoint.com/:f:/g/personal/fadli_aulawi_office_ui_ac_id/Erc9RLcXriNPmTmj8u9FxhcBMpDX5blLuqKsWFPLblu0xA?e=Tvjn9K) that you want to use (EMOS, EPM, or Tafeng). Extract it in its respective directory so the structure will be look like this: (example for EMOS)

```
.
├── EMOS 
│   ├── data
│   ├   ├── data_EMOS-prep // generated directory 
│   ├   ├── EMOS-prep.txt // preprocessed data
│   ├   └── EMOS.csv // raw data
│   ├── Prep-emos.ipynb 
└── └── TAIWI-emos.ipynb 
```

## Program 

Preprocessing code can be found on `Prep-*.ipynb` (for EMOS and EPM). But it is optional because the preprocessed data already available in the downloaded dataset before. 

Training code can be found on `TAIWI-*.ipynb` and will generate the best model as `best-checkpoint.pth`.

Training code also can be run by executing the script:

```
python script.py
```

## Evaluation

So far, this is the best metrics we can achieve by performing hyperparameter tuning on each dataset, and its comparation with the original data:

| Dataset    | NDCG   | Recall | Precision |
|---------   |--------|--------|-----------|
| EMOS       | 0.3214 | 0.3738 | 0.1629    |
| EPM        | 0.2551 | 0.3414 | 0.1121    |
| Tafeng*    | 0.1267 | 0.1642 | 0.0671    |
| TaoBao*    | 0.0815 | 0.1190 | 0.0123    |
| Dunnhumny* | 0.1713 | 0.1791 | 0.1214    |

*Obtained from the original paper
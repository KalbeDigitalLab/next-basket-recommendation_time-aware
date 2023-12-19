import wandb
import os

# TODO: Set for reporting
os.environ["WANDB_API_KEY"] = ''

wandb.login()
wandb.init(project='nbr-emos', id='period-8w')

# TODO: Set for period configuration
periode = "3600 * 24 * 7 * 8"

lst = []
with open('nbr/common/constants.py', 'r') as f:
    lst = f.readlines()

with open('nbr/common/constants.py', 'w') as f:
    lst[0] = f"TIME_SCALAR = {periode}\n"
    f.seek(0)
    f.writelines(lst)
    
import sys
sys.path.append("..")
from nbr.preparation import Preprocess, save_split, Corpus
from nbr.trainer import NBRTrainer
from nbr.model import RepurchaseModule
import torch
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

seed = 10
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# TODO: Set for dataset source and preprocessing configuration
filter = True
min_user = 5
min_item = 10

corpus_path = "EMOS/data/"
dataset_name = "EMOS-prep"

preprocessor = Preprocess(corpus_path, dataset_name)
preprocessor.load_data(min_user, min_item, filt=filter)
save_split(corpus_path, dataset_name, preprocessor)

corpus = Corpus(corpus_path, dataset_name)
corpus.load_data()

# TODO: Set for hyperparameter configuration
trainer = NBRTrainer(
    corpus=corpus,
    max_epochs=50,
    topk=10,
    early_stop_num=50
)

slrc_best_params = {'batch_size': 256, 'lr': 0.00011201144001505824, 'l2_reg_coef': 0.00011498224071460201}

params = {
    "model": RepurchaseModule(
        item_num=corpus.n_items,
        avg_repeat_interval=corpus.total_avg_interval
    ),
    "batch_size": slrc_best_params["batch_size"],
    "lr": slrc_best_params["lr"],
    "l2_reg_coef": slrc_best_params["l2_reg_coef"]
}

trainer.init_hyperparams(**params)

trainer.train()

wandb.finish()

Constrained Variational Policy Optimization for Safe Reinforcement Learning
==================================

Please change the environment name `-e` and the path to DT models `--model_dir` as needed.

Run experiments with CVPO:
```
python script/run_jp.py \
-e Safexp-CarButton1-v0 \
-p cvpo \
-es 3 \ 
-s 123
```

Run experiments with Decision Transformer as the guide policy (an example):
```
python script/run_jp.py \
-e Safexp-CarButton1-v0 \
-p cvpo_jp \
-es 3 \
-s 123 \
--use_dt_guide \
--model_dir decision-transformer/gym/wandb/run-20230810_155710-1gi0bods
```

`-s` controls the random seed used by numpy, pytorch, etc.
`-es` controls the environment seed.
We want to use three different `-s` to plot learning curves.
`-es` can be anything for metadrive.
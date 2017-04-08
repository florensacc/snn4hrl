
# How to run `snn4hrl`

Stochastic Neural Networks for Hierarchical Reinforcement Learning (snn4hrl) as presented at ICLR by *Carlos Florensa, Yan Duan, Pieter Abbeel* (https://openreview.net/forum?id=B1oK8aoxe&noteId=B1oK8aoxe)

[Checkout the videos!](http://bit.ly/snn4hrl-videos)

To reproduce the results, you should first have [rllab](https://github.com/rllab/rllab) and Mujoco v1.31 configured. Then, run the following commands in the root folder of `rllab`:

```bash
git submodule add -f https://github.com/florensacc/snn4hrl.git sandbox/snn4hrl
touch sandbox/__init__.py
```

Then you can do the following:
- Train a SNN for the Swimmer environment via `python sandbox/snn4hrl/runs/train_snn.py`
- Look at the visitation plot including the visitations of every latent code in `data/local/egoSwimmer-snn/`
- Train a hierarchical policy on top of that SNN via `python sandbox/snn4hrl/runs/hier-snn-egoSwimmer-gather.py`

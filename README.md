# Generalizing Persistent Advice in Interactive Reinforcement Learning Scenarios

Seminar project for the course [Bio-Inspired Artificial Intelligence](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/teaching/teaching-2021-wise-bioinspired-ai-seminar.html) (WiSe 21/22) at the University of Hamburg.

## Abstract

Reinforcement learning (RL) is widely used to learn optimal policies in complex environments, but current methods require vast amounts of data and computation. Interactive reinforcement learning (IntRL) methods are researched extensively to leverage expert assistance in reducing this cost. Previous approaches in interactive RL have mainly focused on real-time advice, which is neither subsequently stored nor used for states other than the one it was provided for. Later approaches such as rule-based persistent interactive RL and broad-persistent advising have incorporated methods to retain and reuse knowledge, enabling agents to utilize the general advice in different relevant states. This implementation paper introduces a generalization model for broad-persistent advising which uses bisimulation metrics to evaluate similarity of states and to determine the relevant target state set for given advice. In contrast to direct state-space clustering (e.g. using K-means), bisimulation metrics capture behavioral information of an agent, possibly providing better grouping of similar states. During training of the agent, a state representation space is learned such that the distance between embeddings approximates the bisimulation metric.


## How to run

### Clone the repository
```bash
git clone git@github.com:iibrahimli/gpairls.git
```

### Install dependencies
This can be done by running the following command (better to create a virtual environment):

```bash
cd gpairls
python3 -m pip install -r requirements.txt
```

### Run training
The configuration parameters are in file `gpairls/config.py`, feel free to view/change them. The training script is `gpairls/train.py`. It can be run as such:

```bash
python3 gpairls/train.py
```

After the training is finished, the model and stats such as episode reward and episode length are saved in the directory `logs`. To plot the training stats, run the following command:

```bash
python3 gpairls/plot_stats.py logs/RUN_NAME/stats.csv
```

where `RUN_NAME` must be replaced by the (long) name of the training run, it can be found by `ls`-ing the directory `logs`.


## Acknowledgements

Work was conducted under supervision and advice of [Dr. Francisco Cruz](http://www.franciscocruz.cl).

This codebase is partially based on the following resources:

- Deep Bisimulation for Control (licensed under CC-BY-NC 4.0):
  
  [repository](https://github.com/facebookresearch/deep_bisim4control) | [paper](https://arxiv.org/abs/2006.10742)
- Deep Reinforcement Learning and Interactive Feedback on Robot Domain:
  
  [repository](https://github.com/mwizard1010/robot-control) | [paper](https://arxiv.org/pdf/2110.08003.pdf)

Sources and any modifications on original code are provided in the beginning of relevant files.
## SAC (Soft Actor Critic) - Pytorch (wip)

Implementation of Soft Actor Critic and some of its improvements in Pytorch. Interest comes from watching <a href="https://www.youtube.com/watch?v=17NrtKHdPDw">this lecture</a>

```python
import torch
from SAC_pytorch import (
  SAC,
  Actor,
  Critic,
  MultipleCritics
)

critic1 = Critic(
  dim_state = 5,
  num_cont_actions = 2,
  num_discrete_actions = (5, 5),
  num_quantiles = 3
)

critic2 = Critic(
  dim_state = 5,
  num_cont_actions = 2,
  num_discrete_actions = (5, 5),
  num_quantiles = 3
)

actor = Actor(
  dim_state = 5,
  num_cont_actions = 2,
  num_discrete_actions = (5, 5)
)

agent = SAC(
  actor = actor,
  critics = [
    dict(dim_state = 5, num_cont_actions = 2, num_discrete_actions = (5, 5)),
    dict(dim_state = 5, num_cont_actions = 2, num_discrete_actions = (5, 5)),
  ],
  quantiled_critics = False
)

state = torch.randn(3, 5)
cont_actions, discrete, cont_logprob, discrete_logprob = actor(state, sample = True)

agent(
  states = state,
  cont_actions = cont_actions,
  discrete_actions = discrete,
  rewards = torch.randn(1),
  done = torch.zeros(1).bool(),
  next_states = state + 1
)
```

## Citations

```bibtex
@article{Haarnoja2018SoftAA,
    title   = {Soft Actor-Critic Algorithms and Applications},
    author  = {Tuomas Haarnoja and Aurick Zhou and Kristian Hartikainen and G. Tucker and Sehoon Ha and Jie Tan and Vikash Kumar and Henry Zhu and Abhishek Gupta and P. Abbeel and Sergey Levine},
    journal = {ArXiv},
    year    = {2018},
    volume  = {abs/1812.05905},
    url     = {https://api.semanticscholar.org/CorpusID:55703664}
}
```

```bibtex
@article{Hiraoka2021DropoutQF,
    title   = {Dropout Q-Functions for Doubly Efficient Reinforcement Learning},
    author  = {Takuya Hiraoka and Takahisa Imagawa and Taisei Hashimoto and Takashi Onishi and Yoshimasa Tsuruoka},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2110.02034},
    url     = {https://api.semanticscholar.org/CorpusID:238353966}
}
```

```bibtex
@inproceedings{ObandoCeron2024MixturesOE,
    title   = {Mixtures of Experts Unlock Parameter Scaling for Deep RL},
    author  = {Johan S. Obando-Ceron and Ghada Sokar and Timon Willi and Clare Lyle and Jesse Farebrother and Jakob Foerster and Gintare Karolina Dziugaite and Doina Precup and Pablo Samuel Castro},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:267637059}
}
```

```bibtex
@inproceedings{Kumar2023MaintainingPI,
    title   = {Maintaining Plasticity in Continual Learning via Regenerative Regularization},
    author  = {Saurabh Kumar and Henrik Marklund and Benjamin Van Roy},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:261076021}
}
```

```bibtex
@inproceedings{Kuznetsov2020ControllingOB,
    title   = {Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics},
    author  = {Arsenii Kuznetsov and Pavel Shvechikov and Alexander Grishin and Dmitry P. Vetrov},
    booktitle = {International Conference on Machine Learning},
    year    = {2020},
    url     = {https://api.semanticscholar.org/CorpusID:218581840}
}
```

```bibtex
@article{Zagoruyko2017DiracNetsTV,
    title   = {DiracNets: Training Very Deep Neural Networks Without Skip-Connections},
    author={Sergey Zagoruyko and Nikos Komodakis},
    journal = {ArXiv},
    year    = {2017},
    volume  = {abs/1706.00388},
    url     = {https://api.semanticscholar.org/CorpusID:1086822}
}
```

```bibtex
@article{Abbas2023LossOP,
    title  = {Loss of Plasticity in Continual Deep Reinforcement Learning},
    author = {Zaheer Abbas and Rosie Zhao and Joseph Modayil and Adam White and Marlos C. Machado},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2303.07507},
    url     = {https://api.semanticscholar.org/CorpusID:257504763}
}
```

```bibtex
@article{Zhang2024ReLU2WD,
    title   = {ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs},
    author  = {Zhengyan Zhang and Yixin Song and Guanghui Yu and Xu Han and Yankai Lin and Chaojun Xiao and Chenyang Song and Zhiyuan Liu and Zeyu Mi and Maosong Sun},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.03804},
    url     = {https://api.semanticscholar.org/CorpusID:267499856}
}
```

```bibtex
@inproceedings{Lee2024SimBaSB,
    title  = {SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning},
    author = {Hojoon Lee and Dongyoon Hwang and Donghu Kim and Hyunseung Kim and Jun Jet Tai and Kaushik Subramanian and Peter R. Wurman and Jaegul Choo and Peter Stone and Takuma Seno},
    year   = {2024},
    url    = {https://api.semanticscholar.org/CorpusID:273346233}
}
```

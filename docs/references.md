# References

Works this project builds on, compares with or uses. Grouped by topic; each entry says why it matters here.

## Closest prior work

- Li & Sun, *Delay Randomization Widens the Delay Margin of Learned Humanoid Motion Trackers*,
  arXiv 2610.32277, 2026. [link](https://arxiv.org/abs/2610.32277). Humanoid tracker, action delay
  U[0, 50 ms]; the delay arm is fine-tuned from a no-delay policy; margin widens from 20–30 to
  50–70 ms. Our "delay margin" metric is theirs. Our H1/H2 replicate their result on a quadruped;
  the history, memory, oracle and probing comparisons are new here.

## Delay in legged-robot sim-to-real

- Tan et al., *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots*, RSS 2018.
  [arXiv 1804.10332](https://arxiv.org/abs/1804.10332). Latency as a main cause of the reality gap.
- Imai et al., *Vision-Guided Quadrupedal Locomotion in the Wild with Multi-Modal Delay
  Randomization*, IROS 2022. [arXiv 2109.14549](https://arxiv.org/abs/2109.14549). Proprioceptive
  delay randomized over 0–40 ms; fixed-delay policies degrade at other delays.
- Margolis & Agrawal, *Walk These Ways*, CoRL 2022.
  [code](https://github.com/Improbable-AI/walk-these-ways). Go1 with a constant ≈ 30 ms lag and a
  15-frame observation history.
- Zakka et al., *MuJoCo Playground*, RSS 2025. [arXiv 2502.08844](https://arxiv.org/abs/2502.08844).
  The Go1 environment and PPO setup this project starts from (no latency modelled).
- Xie et al., *Dynamics Randomization Revisited*, ICRA 2021.
  [project page](https://www.pair.toronto.edu/understanding-dr/).
- Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots*, Science Robotics
  2019. [arXiv 1901.08652](https://arxiv.org/abs/1901.08652). Learned actuator model; our
  first-order motor lag is a simpler stand-in.
- Siekmann et al., *Learning Memory-Based Control for Human-Scale Bipedal Locomotion*, RSS 2020.
  [arXiv 2006.02402](https://arxiv.org/abs/2006.02402). Recurrent policies that transfer to hardware.

## Delay in reinforcement learning

- Katsikopoulos & Engelbrecht, *Markov Decision Processes with Delays and Asynchronous Cost
  Collection*, IEEE TAC 48(4), 2003. [doi:10.1109/TAC.2003.809799](https://doi.org/10.1109/TAC.2003.809799).
  A constant delay is handled by adding the pending actions to the state: the basis for the
  history method (D); observation and action delay are equivalent (H5).
- Walsh, Nouri, Li & Littman, *Learning and Planning in Environments with Delayed Feedback*,
  Autonomous Agents and Multi-Agent Systems, 2009. (year and venue not re-verified at closing; an earlier
  conference version may exist).
- Ramstedt & Pal, *Real-Time Reinforcement Learning*, NeurIPS 2019.
  [arXiv 1911.04448](https://arxiv.org/abs/1911.04448).
- Derman, Dalal & Mannor, *Acting in Delayed Environments with Non-Stationary Markov Policies*,
  ICLR 2021. [arXiv 2101.11992](https://arxiv.org/abs/2101.11992).
- Wikman et al., *Adaptive Reinforcement Learning for Unobservable Random Delays*, ICML 2026.
  [poster](https://icml.cc/virtual/2026/poster/64105).

## Memory and probing

- Chamachot, *Evaluating Factor-Wise Auxiliary Dynamics Supervision for Latent Structure and
  Robustness in Simulated Humanoid Locomotion*, arXiv 2603.21268, 2026.
  [link](https://arxiv.org/abs/2603.21268). Its main model is a transformer with a factored latent;
  the result used here is from its LSTM baseline, which held almost no delay information
  (R² ≈ 0.01) when five factors vary; our GRU encodes it strongly (R² 0.78) when only delay varies.

## Methods and tools

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
  [arXiv 1707.06347](https://arxiv.org/abs/1707.06347).
- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS
  2021. [arXiv 2108.13264](https://arxiv.org/abs/2108.13264). IQM and stratified bootstrap CIs (rliable).
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2022.
  [legged_gym](https://github.com/leggedrobotics/legged_gym). Fall/termination conventions.
- Todorov, Erez & Tassa, *MuJoCo: A Physics Engine for Model-Based Control*, IROS 2012; MJX and
  Brax (Freeman et al., 2021) for GPU simulation and the reference PPO.

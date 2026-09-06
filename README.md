# 심층강화학습 심화 — Deep Reinforcement Learning

<p class="gh-only">📖 <b>웹 교재로 보기:</b> <a href="https://ai-boostcamp.github.io/ReinforcementLearning/">https://ai-boostcamp.github.io/ReinforcementLearning/</a></p>

**표(table) 기반 강화학습의 원리에서 시작해, 신경망을 결합한 심층강화학습(DQN · Policy
Gradient · DDPG)과 그 약점을 고친 최신 알고리즘(DDQN · TD3 · A3C · PPO · SAC)까지** 다루는
교재다. 강의 "Deep Reinforcement Learning(심층강화학습)"(홍근선, ㈜한국AI연구소)의
슬라이드를 읽는 글로 다시 쓴 것으로, 모든 실습은 Google Colab 노트북(Keras/TensorFlow, 일부
PyTorch 이식판 병기)과 함께 제공된다.

2장에서는 MDP 와 Bellman 방정식, 동적 계획법, Monte-Carlo, TD, SARSA, Q-Learning 을 Gridworld
· CliffWalking 위에서 익힌다. 3장에서는 함수 근사와 신경망을 결합해 Naïve Deep Q · DQN ·
REINFORCE · Actor-Critic · DDPG 를 CartPole · Pendulum 에서 구현한다. 4장에서는 프로젝트
준비(MDP 설계 · 재현성)를 짚은 뒤 DDQN · TD3 · A3C · PPO · SAC 를 CartPole · LunarLander 에서
구현하고, 부록에서 그 이후의 알고리즘과 연구 동향을 훑는다. 장 번호는 강의 슬라이드를 그대로
따르며, **1장(Deep Learning 개념 정리)은 목차만 남기고 본문은 싣지 않았다.**

## 목차

**[교재 개요와 전체 목차](00_교재개요.md)** — 강의 전체 목차, 이 교재의 범위, 실습 번호 규칙

**1장 Deep Learning 개념 정리** — [장 개요(목차만)](ch01_DL개념정리/00_1장개요_장개요.md)

**2장 강화학습(Reinforcement Learning)** — [장 개요](ch02_강화학습/00_2장개요_장개요.md)

- [2.1 강화학습 개요](ch02_강화학습/2.1_강화학습개요.md) — 연구 · 투자의 흐름, 기계학습의 세
  갈래와 강화학습, 구성요소와 어려움, 주요 알고리즘의 발전과 분류, 적용 사례
- [2.2 MDP 와 Bellman Equation](ch02_강화학습/2.2_MDP와Bellman.md) — Markov Process · Reward
  Process · Decision Process, 상태 · 행동 가치 함수, Bellman 기대 · 최적 방정식, MDP 환경 구축 실습
- [2.3 Dynamic Programming](ch02_강화학습/2.3_DP.md) — 정책 평가 · 정책 개선 · 정책 반복 · 가치
  반복, 비동기 DP, Gridworld 실습
- [2.4 Monte-Carlo Method](ch02_강화학습/2.4_MonteCarlo.md) — Model-free 예측과 제어, First/Every
  visit, ε-greedy, GLIE, Gridworld 실습
- [2.5 Temporal-Difference Method](ch02_강화학습/2.5_TD.md) — TD(0) · n-step · TD(λ),
  bias-variance, MC 와의 비교 실습
- [2.6 SARSA Algorithm](ch02_강화학습/2.6_SARSA.md) — On-policy TD 제어, CliffWalking 실습
- [2.7 Off-policy Control 과 Q-Learning](ch02_강화학습/2.7_QLearning.md) — Importance sampling,
  Q-Learning, Maximization bias 와 Double Q-Learning, SARSA 와의 비교 실습

**3장 심층강화학습(Deep Reinforcement Learning)** — [장 개요](ch03_심층강화학습/00_3장개요_장개요.md)

- [3.1 Function Approximation](ch03_심층강화학습/3.1_FunctionApproximation.md) — 표에서 함수로,
  가치 함수 근사와 gradient
- [3.2 Deep Neural Network](ch03_심층강화학습/3.2_DNN.md) — 회귀 · 분류 모델을 가치 · 정책 함수로
- [3.3 Naïve Deep Q-Learning](ch03_심층강화학습/3.3_NaiveDeepQ.md) — Q-Network 로 CartPole, 왜
  불안정한가
- [3.4 DQN](ch03_심층강화학습/3.4_DQN.md) — Experience Replay · Target Network, CartPole 실습,
  Atari Breakout 참고 실습
- [3.5 Policy Gradient](ch03_심층강화학습/3.5_PolicyGradient.md) — Policy Gradient Theorem,
  REINFORCE 와 baseline, Return Normalization, W&B logging, TD Actor-Critic(A2C)
- [3.6 DDPG](ch03_심층강화학습/3.6_DDPG.md) — Deterministic Policy Gradient, Actor-Critic +
  Replay + Target + Soft update, Pendulum · LunarLander 실습

**4장 Advanced Deep RL** — [장 개요](ch04_AdvancedDeepRL/00_4장개요_장개요.md)

- [4.1 Deep RL Project 준비](ch04_AdvancedDeepRL/4.1_프로젝트준비.md) — MDP 설계, 알고리즘과
  hyper parameter 선택, 라이브러리, 성능 평가의 통계적 함정
- [4.2 DDQN(Double DQN)](ch04_AdvancedDeepRL/4.2_DDQN.md) — Overestimation bias 와 Double
  Q-learning + DQN, CartPole 실습
- [4.3 TD3(Twin Delayed DDPG)](ch04_AdvancedDeepRL/4.3_TD3.md) — Clipped Double Q · Delayed
  policy update · Target policy smoothing, LunarLander 실습, shared critic · ablation 과제
- [4.4 A3C](ch04_AdvancedDeepRL/4.4_A3C.md) — 분산 · 병렬 학습, 비동기 actor-learner, Entropy
  regularization, Python multiprocessing 으로 CartPole 실습
- [4.5 PPO](ch04_AdvancedDeepRL/4.5_PPO.md) — CPI → TRPO → PPO, KL divergence, Clipped surrogate
  loss, GAE, CartPole 실습
- [4.6 SAC](ch04_AdvancedDeepRL/4.6_SAC.md) — Maximum Entropy RL, soft value · soft policy
  iteration, Squashed Gaussian 과 reparameterization, temperature 학습(SAC-v2), LunarLander 실습

**부록**

- [부록 A. 최신 알고리즘과 연구 동향](부록A_Appendix/부록A_Appendix.md) — GRPO, Ape-X, Noisy
  Net, Rainbow, FinRL, Spinning Up 벤치마크, 표현학습 · 일반화 · Offline · Multi-Task ·
  Multi-Agent · Evolution Strategies, Deep RL 분류표

## 실습 안내

각 실습 제목 아래의 **"Open in Colab" 배지**를 누르면 노트북이 Colab 에서 바로 열린다.
노트북 파일은 이 저장소의 [`code/`](https://github.com/AI-BoostCamp/ReinforcementLearning/tree/main/code)
폴더에 교재의 실습 번호와 같은 이름으로 들어 있다(`실습 2.3.1` ↔ `code/2.3.1.DP_Policy_Iteration.ipynb`).
본문 코드는 Keras/TensorFlow 판이고, 같은 실습의 **PyTorch 이식판**(`*.pt.*.ipynb`)은 실습
머리에 배지로 함께 걸어 두었다.

실행 전 준비:

- **`lib/` 폴더** — 2장의 Gridworld 노트북과 여러 실습이 `code/lib/`(`gridworld.py`,
  `variable_print.py`, `render.py` 등)를 쓴다. 노트북은 Google Drive 의
  `내 드라이브/Colab Notebooks/lib` 를 `sys.path` 에 넣으므로, [`code/lib/`](https://github.com/AI-BoostCamp/ReinforcementLearning/tree/main/code/lib)
  폴더를 내 Drive 의 `Colab Notebooks/lib` 에 복사해 두고 노트북 첫 셀에서 Drive 를 mount 한다.
  학습된 모델을 저장하는 실습은 같은 곳의 `Colab Notebooks/models/DRL/` 폴더를 쓴다.
- **환경 점검** — [`code/0.0.test_RL.ipynb`](code/0.0.test_RL.ipynb) 로 Colab VM 의 Python ·
  TensorFlow · Gym 버전과 렌더링 도구를 먼저 확인한다.
- **런타임** — 2장과 3장 초반은 CPU 로 충분하다. 3.4 의 Atari Breakout, 4장의 LunarLander
  실습(TD3 · SAC)은 GPU 런타임에서 훨씬 빠르다. A3C 실습은 고용량 RAM 런타임에서 CPU 4 개를
  쓴다.
- **Gym 과 Gymnasium** — 노트북마다 `gym`(구 API, `env.step()` 이 4 개 반환)과
  `gymnasium`(신 API, 5 개 반환)이 섞여 있다. 각 절의 본문이 어느 쪽인지 적어 두었으니,
  버전 오류가 나면 반환값 개수부터 확인한다.
- **Weights & Biases** — 3.5 절의 실습 3.5.3(Model Logging)만 W&B 계정과 API 키가 필요하다.
  키는 노트북에 적지 말고 Colab 보안 비밀(secrets)이나 `wandb login` 으로 입력한다.

## 저자

**홍근선** — ㈜한국AI연구소 대표이사 · 전자계산기 기술사 · 성균관대학교 겸임교수 (gshong@ai-camp.kr)

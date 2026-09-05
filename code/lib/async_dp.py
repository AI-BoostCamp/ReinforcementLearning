import numpy as np
from queue import PriorityQueue


class AsyncDP:

    def __init__(self,
                 gamma=1.0,
                 error_tol=1e-8):
        self.gamma = gamma
        self.error_tol = error_tol

        # Following attributes will be set after call "set_env()"

        self.env = None  # environment
        self.policy = None  # policy
        self.ns = None  # Num. states
        self.na = None  # Num. actions
        self.P = None  # Transition tensor
        self.R = None  # Reward tensor

    def set_env(self, env, policy=None):
        self.env = env
        if policy is None:
            self.policy = np.ones([env.nS, env.nA]) / env.nA

        self.ns = env.nS
        self.na = env.nA
        self.P = env.P_tensor  # Rank 3 tensor [num. actions x num. states x num. states]
        self.R = env.R_tensor  # Rank 2 tensor [num. actions x num. states]

        print("Asynchronous DP initialized")
        print("Environment spec:  Num. state = {} | Num. actions = {} ".format(env.nS, env.nA))

    def compute_q_from_v(self, value):
        return self.R.T + self.gamma * self.P.dot(value)  # [num. actions x num. states]

    def construct_policy_from_v(self, value):
        qs = self.compute_q_from_v(value)  # [num. actions x num. states]

        # construct greedy policy from Qs.
        pi = np.zeros_like(self.policy)
        pi[np.arange(qs.shape[1]), qs.argmax(axis=0)] = 1
        return pi

    def in_place_vi(self, v_init=None):

        if v_init is not None:
            value = v_init
        else:
            value = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['gap'] = list()
        info['converge'] = False
        info['step'] = None

        steps = 0
        while True:
            # perform in-place VI
            delta_v = 0
            for s in range(self.ns): # 각각의 s에 대해서 v계산: memory 절감
                # qs(a,) <- self.R.T + self.gamma * self.P.dot(value)
                qs = self.compute_q_from_v(value)[:, s] 
                v = qs.max(axis=0)  # v <- max_a(Q) 
                # v 변화량 accum 
                delta_v += np.linalg.norm(value[s] - v)
                value[s] = v
            info['v'].append(value.copy())
            # v로 qs구하고, greedy improvement : policy improvement와 동일  
            pi = self.construct_policy_from_v(value)
            info['pi'].append(pi)
            info['gap'].append(delta_v)
            print(f"iter[{steps:02d}] Delta V: {delta_v:.5f}")

            if delta_v < self.error_tol:
                if info['converge']:
                    info['step'] = steps
                    break
                else:
                    info['converge'] = True
            else:
                steps += 1
        return info

    def prioritized_sweeping_vi(self, v_init=None):
 
        if v_init is not None:
            value = v_init
        else:
            value = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['gap'] = list()
        info['converge'] = False
        info['step'] = None

        steps = 0
        while True:
            # compute the Bellman errors
            # e(s,) <- v(s,) - v(s',)
            bellman_errors = value - (self.R.T + self.P.dot(value)).max(axis=0)
            state_indices = range(self.ns)
            # priority queue <- (-error,s_idx)
            priority_queue = PriorityQueue()
            for bellman_error, s_idx in zip(bellman_errors, state_indices):
                priority_queue.put((-bellman_error, s_idx))
            # pi update 
            pi = self.construct_policy_from_v(value)
            info['pi'].append(pi.copy())
            # error가 큰것부터 v update 
            delta_v = 0
            while not priority_queue.empty():
                be, s = priority_queue.get()
                qs = self.compute_q_from_v(value)[:, s]
                v = qs.max(axis=0)  # get max value along the actions
                delta_v += np.linalg.norm(value[s] - v)
                value[s] = v
            info['gap'].append(delta_v)
            info['v'].append(value.copy())
            print(f"iter[{steps:02d}] Delta V: {delta_v:.5f}")
            if delta_v < self.error_tol:
                if info['converge']:
                    info['step'] = steps
                    break
                else:
                    info['converge'] = True
            else:
                steps += 1
        return info

    def in_place_vi_partial_update(self,
                                   v_init=None,
                                   update_prob=0.5,
                                   vi_iters: int = 100):

        if v_init is not None:
            value = v_init
        else:
            value = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['gap'] = list()

        for steps in range(vi_iters):
            # perform in-place VI
            delta_v = 0
            for s in range(self.ns): # in_place VI(partial sweeping)
                perform_update = np.random.binomial(size=1, n=1, p=update_prob)
                if not perform_update: continue
                # qs(a,) <- self.R.T + self.gamma * self.P.dot(value)
                qs = self.compute_q_from_v(value)[:, s]
                v = qs.max(axis=0)  # v <- max_a(Q) 
                # v 변화량 accum 
                delta_v += np.linalg.norm(value[s] - v)
                value[s] = v
            info['gap'].append(delta_v)
            info['v'].append(value.copy())
            pi = self.construct_policy_from_v(value)
            info['pi'].append(pi)
            print(f"iter[{steps:02d}] Delta V: {delta_v}")

        return info

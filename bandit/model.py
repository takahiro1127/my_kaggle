import numpy as np
import random
# 今回はtompson samplingで作る
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0
#一番効率の良い、トンプソンサンプリング
class ThompsonSampling():
#values: 報酬の配列
#
    def __init__(self, counts_alpha, counts_beta, values):
        self.counts_alpha = counts_alpha
        self.counts_beta = counts_beta
        self.alpha = 1
        self.beta = 1
        self.values = values

    def initialize(self, n_arms):
        self.counts_alpha = np.zeros(n_arms)
        self.counts_beta = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
#n_armsは選択肢の数

    def select_arm(self):
        theta = [(arm, random.betavariate(self.counts_alpha[arm] + self.alpha,　self.counts_beta[arm] + self.beta)) for arm in range(len(self.counts_alpha))]
        theta = sorted(theta, key=lambda x: x[1])
        return theta[-1][0]

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1
        n = float(self.counts_alpha[chosen_arm]) + self.counts_beta[chosen_arm]
        self.values[chosen_arm] = (n - 1) / n * self.values[chosen_arm] + 1 / n * reward

# 一番簡単なe-greedy
class EpsilonGreedy():
    # epsilon: 探索する確率
    # count: 各候補の探索回数
    # value: 報酬の平均値
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, len(self.values) - 1)
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
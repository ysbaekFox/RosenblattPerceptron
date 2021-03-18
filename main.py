"""
-----------------------
# RosenblattPerceptron
# Created by ysbaekFox
# Date : 2021.03.16
# brief :
-----------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        :param eta: 학습률
        :param n_iter: 에포크수 (학습 횟수)
        :param random_state: 가중치의 초기 값 (무작위 값)
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        :param X: Data Set, [n_samples, n_features]
        :param y: 실제 값, list 형태 (vector) Target Value
        :return:
        """

        # np.random.RandomState(seed), seed 값을 다르게 해주어야 동일한 코드 내에서 다른 무작위 값을 얻을 수 있음.
        rgen = np.random.RandomState(self.random_state)
        # 평균이 0.0, 표준편차가 0.01인 정규분포에서 size 수 만큼 난수 생성.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                if 0.0 != update:
                    errors += 1
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # 벡터의 점곱 (내적)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # 삼항 연산
        return np.where(self.net_input(X=X) >= 0.0, 1, -1)


if __name__ == '__main__':

    df = pd.read_csv('iris.data')

    # 1, 3, 5번 행 데이터 출력
    # print(df.loc[[1, 3, 5]])
    # 0번 행 데이터 출력
    # print(df.iloc[0])
    # 1, 3, 5번 행 데이터 출력
    # print(df.iloc[[1, 3, 5]])
    # 즉, 행 데이터만 필터링 할때는 iloc와 loc가 동일하다

    # 0번 행부터~마지막행까지, column Name이 '0', '2', '4'인 것들로 필터링
    # print(df.loc[0:, ['0', '2', '4']])
    # 0번 행부터~마지막행가지, 1번 index column 부터 3(<4) 번 inedx column 까지 필터링
    # print(df.iloc[0:, 1:4])

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    # scatter
    # scatter(X[:50, 0], X[:50, 1])
    # X[:50, 0] --> list X 0 <= value < 50이면서 0번째

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
    
    # 퍼셉트론 훈련
    # 학습률과 에포크 횟수 설정
    ppn = Perceptron(eta=0.1, n_iter=50)
    # 학습 시작 X : dataset, y : 실제 값
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of errors')
    plt.show()


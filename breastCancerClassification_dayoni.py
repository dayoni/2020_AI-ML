import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.optimize import minimize

data = np.loadtxt(fname="Breast_cancer_data.txt")  # 데이터 생성

X = data[:, :9]  # 입력, 0열부터 8열까지
T = data[:, 9]  # 출력, 9열만
X = X.astype(int)  # 실수형으로 들어온 데이터를 정수형으로 변환
T = T.astype(int)  # 실수형으로 들어온 데이터를 정수형으로 변환

X_n = data.shape[0]  # data의 행의 갯수를 X_n에 대입


def logistic9(x0, x1, x2, x3, x4, x5, x6, x7, x8, w):  # 로지스틱 회귀 모델 생성
    y = 1 / (1 + np.exp(-((w[0] * x0) + (w[1] * x1) + (w[2] * x2) + (w[3] * x3) + (w[4] * x4)
                          + (w[5] * x5) + (w[6] * x6) + (w[7] * x7) + (w[8] * x8) + w[9])))
    return y


def cee_logistic9(w, x, t):  # 교차 엔트로피 오차(비용함수) 구하기
    y = logistic9(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee


def dcee_logistic9(w, x, t):
    y = logistic9(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], w)
    dcee = np.zeros(10)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n]) * x[n, 2]
        dcee[3] = dcee[3] + (y[n] - t[n]) * x[n, 3]
        dcee[4] = dcee[4] + (y[n] - t[n]) * x[n, 4]
        dcee[5] = dcee[5] + (y[n] - t[n]) * x[n, 5]
        dcee[6] = dcee[6] + (y[n] - t[n]) * x[n, 6]
        dcee[7] = dcee[7] + (y[n] - t[n]) * x[n, 7]
        dcee[8] = dcee[8] + (y[n] - t[n]) * x[n, 8]
        dcee[9] = dcee[9] + (y[n] - t[n])
    dcee = dcee / X_n
    return dcee


W = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dcee_logistic9(W, X, T)


def fit_logistic9(w_init, x, t):
    res = minimize(cee_logistic9, w_init, args=(x, t),
                   jac=dcee_logistic9, method="CG")
    return res.x


W_init = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
W = fit_logistic9(W_init, X, T)
print(
    "w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}, w3 = {3:.2f}, w4 = {4:.2f},      w5 = {5:.2f}, w6 = {6:.2f}, w7 = {7:.2f}, w8 = {8:.2f}, w9 = {9:.2f}".format(
        W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7], W[8], W[9]))


def kfold_func(w_k, x, t, k):  # x=측정데이터, t=결과데이터, k=몇개로 나눌것인가?
    n = x.shape[0]
    cee_train = np.zeros(k)
    cee_test = np.zeros(k)

    for i in range(0, k):  # i = 0, 1, 2 ... k-1
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]
        wm = fit_logistic9(w_k, x_train, t_train)
        cee_train[i] = cee_logistic9(wm, x_train, t_train)
        cee_test[i] = cee_logistic9(wm, x_test, t_test)
    return cee_train, cee_test


W_K = fit_logistic9(W_init, X, T)
K = 40
kfold_func(W_K, X, T, K)


def validate_model(w):
    test = np.loadtxt(fname="Sample_test_data.txt")
    X = test[:, :9].astype(int)
    T = test[:, 9].astype(int)

    N = X.shape[0]

    y = np.zeros(N)
    decision = np.zeros(N).astype(int)
    err_cnt = 0

    print('No. \t Y \t T')
    print('--------------------')
    for i in range(N):
        x = np.r_[X[i, :], 1]
        u = np.array(w).dot(x)
        y[i] = 1 / (1 + np.exp(-u))
        if y[i] < 0.5:
            decision[i] = 1

        if decision[i] != T[i]:
            err_cnt = err_cnt + 1

        print('{0} \t {1} \t {2}'.format(i, decision[i], T[i]))

    hit_ratio = np.round((1 - err_cnt / N) * 100, 1)

    print('--------------------')
    print('Total error : {0} out of [1]'.format(err_cnt, N))
    print('Hit_ratio : {0:.1f} %'.format(hit_ratio))

    return hit_ratio


W = np.loadtxt(fname="parameter.txt")

print("W = " + np.str(W))
print("\n")

validate_model(W)
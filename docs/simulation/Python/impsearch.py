from time import time
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from abess.linear import LogisticRegression
from abess.datasets import make_glm_data

np.random.seed(0)

n = 500
p = 2000
k = 20
rho = 0.1
M = 50
search_path = [32, 64, 128, 256, 512, 1024, 2048]

met_save = True
res_save = True
figure_save = True

met = np.zeros((len(search_path), M, 2))
res = np.zeros((len(search_path), 5))
for m in range(M):
    train = make_glm_data(n=n, p=p, k=k, family='binomial')
    test = make_glm_data(n=n, p=p, k=k, family='binomial', coef_=train.coef_)

    print("==> Iter : ", m)

    for i, imp in enumerate(search_path):
        ts = time()
        model = LogisticRegression(
            support_size=range(100),
            important_search=imp)
        model.fit(train.x, train.y)
        te = time()
        met[i, m, 0] = roc_auc_score(test.y, model.predict(test.x))
        met[i, m, 1] = te - ts

for i, imp in enumerate(search_path):
    res[i, 0] = imp
    m = met[i].mean(axis=0)
    se = met[i].std(axis=0) / np.sqrt(M - 1)
    res[i, 1:5] = np.hstack((m, se))

if met_save:
    np.save('met.npy', met)

if res_save:
    np.save('res.npy', res)

if figure_save:
    res = np.load("res.npy")
    # print(res)

    plt.figure(figsize=(20, 6))

    plt.subplot(121)
    plt.errorbar(res[:, 0], res[:, 1], yerr=res[:, 3] * 2, capsize=3)
    plt.xticks(res[:, 0], [str(i) for i in res[:, 0]])
    plt.ylim(0.9, 1)
    plt.ylabel('AUC')
    plt.xlabel('log2(important_search)')
    # plt.savefig('./auc.png')

    plt.subplot(122)
    plt.errorbar(res[:, 0], res[:, 2], yerr=res[:, 4] * 2, capsize=3)
    plt.xticks(res[:, 0], [str(i) for i in res[:, 0]])
    plt.title('Time(/s)')
    plt.xlabel('log2(important_search)')
    # plt.savefig('./time.png')

    plt.savefig('./impsearch.png')
    print('Figure saved.')

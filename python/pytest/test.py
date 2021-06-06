import numpy as np
from abess.linear import *
from abess.gen_data import gen_data, gen_data_splicing

if __name__ == "__main__":
    n = 2000
    p = 10
    k = 2
    family = "gaussian"
    rho = 0.5
    sigma = 1
    M = 1
    # data = gen_data_splicing(family=family, n=n, p=p, k=k, rho=rho, M=M)

    s_max = k + 1

    model = abessSPCA(path_type="seq", sequence=range(k, s_max), ic_type='ebic', is_screening=False, screening_size=20,
                    K_max=10, epsilon=10, powell_path=2, s_min=1, s_max=p, lambda_min=0.01, lambda_max=100, is_cv=True, K=5,
                    exchange_num=2, tau=0.01 * np.log(n*p) / n, 
                    primary_model_fit_max_iter=10, primary_model_fit_epsilon=1e-6, early_stop=False, approximate_Newton=True, ic_coef=1., thread=1)
    # model.fit(data.x, data.y)

    np.random.seed(2)
    x1 = np.random.randn(n, 1)
    x1 /= np.linalg.norm(x1)
    test_x = x1.dot(np.random.randn(1, p)) + 0.01 * np.random.randn(n, p)
    test_x = test_x - test_x.mean(axis = 0)
    g_index = list(range(0, p))

    # x1 = np.zeros((10, 5))
    # for i in range(5):
    #     x1[i * 2, i] = 1
    #     x1[i * 2 + 1, i] = -1
    # x1 = np.matrix(x1)

    # test_x = x1[:, 0]
    # g_index = np.zeros(1)
    # for i in range(1, 5):
    #     x2 = x1[:, i].dot( np.ones((1, i+1)) / np.sqrt(i+1) )
    #     test_x = np.hstack((test_x, x2))
    #     g_index = np.append(g_index, np.ones(i+1)*i)

    
    test_y = np.ones(test_x.shape[0])
    
    model.fit(test_x, test_y, is_normal = False, group = g_index)

    print(model.beta)

    print()
    # print(test_x)

    xb = test_x.dot(model.beta)
    explain = xb.T.dot(xb)
    print( explain / sum(np.diagonal(test_x.T.dot(test_x))) )

    print( ' non-zero = ', np.count_nonzero(model.beta) )

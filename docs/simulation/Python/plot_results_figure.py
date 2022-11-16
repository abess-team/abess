import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% run test
os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = [
    'Lm0.1_res.npy',
    'Lm0.7_res.npy',
    'Logistic0.1_res.npy',
    'Logistic0.7_res.npy',
    'Lm0.1_data.npy',
    'Lm0.7_data.npy',
    'Logistic0.1_data.npy',
    'Logistic0.7_data.npy'
]
benchmarks = [
    'run_benchmark_linear.py 0.1',
    'run_benchmark_linear.py 0.7',
    'run_benchmark_logistic.py 0.1',
    'run_benchmark_logistic.py 0.7'
]

python_path = sys.executable
for i in range(4):
    if not (os.path.exists(files[i]) and os.path.exists(files[i]) + 4):
        os.system(python_path + ' ' + benchmarks[i])

# %% load results
lm1_res = np.load('Lm0.1_res.npy')
lm7_res = np.load('Lm0.7_res.npy')
logi1_res = np.load('Logistic0.1_res.npy')
logi7_res = np.load('Logistic0.7_res.npy')

lm1_data = np.load('Lm0.1_data.npy')
lm7_data = np.load('Lm0.7_data.npy')
logi1_data = np.load('Logistic0.1_data.npy')
logi7_data = np.load('Logistic0.7_data.npy')


# %% plot performance results
plt.figure(1,figsize=(18, 27))

# lm
color = ['#00AF91', '#FFCC29', '#5089C6']
c1 = mpatches.Patch(color=color[0], label='Lasso')
c2 = mpatches.Patch(color=color[1], label='OMP')
c3 = mpatches.Patch(color=color[2], label='ABESS')

plt.subplot(321)
for i in range(lm1_data.shape[0]):
    plt.boxplot(x=[lm1_data[i, lm1_data[i, :, 0] < 100, 0],
                   lm7_data[i, lm7_data[i, :, 0] < 100, 0]],
                patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 5], widths=0.7,
                boxprops=dict(facecolor=color[i]))

plt.xlabel('low corr                                         high corr')
plt.title('Linear - Predict Error')

plt.subplot(323)
for i in range(lm1_data.shape[0]):
    plt.boxplot(x=[lm1_data[i, :, 1], lm7_data[i, :, 1]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 5], widths=0.7,
                boxprops=dict(facecolor=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Linear - Coefficient error')

plt.subplot(325)
for i in range(lm1_data.shape[0]):
    plt.boxplot(x=[lm1_data[i, :, 3], lm7_data[i, :, 3]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 5], widths=0.7,
                boxprops=dict(facecolor=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Linear - FPR')
plt.legend(handles=[c1, c2, c3])

# logi
color = ['#00AF91', '#5089C6']

plt.subplot(322)
for i in range(logi1_data.shape[0]):
    plt.boxplot(x=[logi1_data[i, :, 0], logi7_data[i, :, 0]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 4], widths=0.7,
                boxprops=dict(facecolor=color[i]))

plt.xlabel('low corr                                         high corr')
plt.title('Logistic - AUC')

plt.subplot(324)
for i in range(logi1_data.shape[0]):
    plt.boxplot(x=[logi1_data[i, :, 1], logi7_data[i, :, 1]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 4], widths=0.7,
                boxprops=dict(facecolor=color[i]),
                medianprops=dict(color=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Logistic - Coefficient error')

plt.subplot(326)
for i in range(logi1_data.shape[0]):
    plt.boxplot(x=[logi1_data[i, :, 3], logi7_data[i, :, 3]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 4], widths=0.7,
                boxprops=dict(facecolor=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Logistic - FPR')
plt.legend(handles=[c1, c3])

plt.savefig('perform.png')
print('Perfromance figure saved.')

# %% plot timing results
plt.figure(2, figsize=(14, 6))

# lm_time
plt.subplot(121)
color = ['#00AF91', '#FFCC29', '#5089C6']
c1 = mpatches.Patch(color=color[0], label='Lasso')
c2 = mpatches.Patch(color=color[1], label='OMP')
c3 = mpatches.Patch(color=color[2], label='ABESS')

temp = np.vstack((lm1_res[:, [5, 11]], lm7_res[:, [5, 11]]))
plt.bar(x=[1, 2, 3, 5, 6, 7], height=temp[:, 0], yerr=temp[:, 1] * 2,
        capsize=10, tick_label='', color=color)
plt.xlabel('low corr                                           high corr')
plt.title('Linear')
plt.ylabel('time(s)')
plt.ylim((0, 8))
plt.legend(handles=[c1, c2, c3])
# plt.savefig('./lm_time.png')

# logi_time

plt.subplot(122)
color = ['#00AF91', '#5089C6']

temp = np.vstack((logi1_res[:, [5, 11]], logi7_res[:, [5, 11]]))
lm_time = plt.bar(x=[1, 2, 4, 5], height=temp[:, 0], yerr=temp[:, 1] * 2,
                  capsize=10, tick_label='', color=color[0:2])
plt.title('Logistic')
plt.ylabel('time(s)')
plt.ylim((0, 8))
plt.legend(handles=[c1, c3])
plt.xlabel('low corr                                                high corr')
# plt.savefig('./logi_time.png')

plt.savefig('timings.png')
print('Timing figure saved.')

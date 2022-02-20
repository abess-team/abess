import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# run = ''
# while run != 'y' and run != 'n':
#     run = input('Re-run the simulation? (y/n/c) ')
#     if run == 'c':
#         os._exit(0)
#     elif run == 'y':
#         os.system('python linear_test.py 0.1')
#         os.system('python linear_test.py 0.7')
#         os.system('python logi_test.py 0.1')
#         os.system('python logi_test.py 0.7')
#         break

python_path = sys.executable
os.system(python_path + ' ./linear_test.py 0.1')
os.system(python_path + ' ./linear_test.py 0.7')
os.system(python_path + ' ./logi_test.py 0.1')
os.system(python_path + ' ./logi_test.py 0.7')

# simulation results
lm1 = np.load('./Lm0.1_data.npy')
lm7 = np.load('./Lm0.7_data.npy')
logi1 = np.load('./Logistic0.1_data.npy')
logi7 = np.load('./Logistic0.7_data.npy')

plt.figure(figsize=(18, 27))

# lm
color = ['#00AF91', '#FFCC29', '#5089C6']
c1 = mpatches.Patch(color=color[0], label='Lasso')
c2 = mpatches.Patch(color=color[1], label='OMP')
c3 = mpatches.Patch(color=color[2], label='ABESS')

plt.subplot(321)
for i in range(lm1.shape[0]):
    plt.boxplot(x=[lm1[i, lm1[i, :, 0] < 100, 0],
                   lm7[i, lm7[i, :, 0] < 100, 0]],
                patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 5], widths=0.7,
                boxprops=dict(facecolor=color[i]))

plt.xlabel('low corr                                         high corr')
plt.title('Linear - Predict Error')

plt.subplot(323)
for i in range(lm1.shape[0]):
    plt.boxplot(x=[lm1[i, :, 1], lm7[i, :, 1]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 5], widths=0.7,
                boxprops=dict(facecolor=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Linear - Coefficient error')

plt.subplot(325)
for i in range(lm1.shape[0]):
    plt.boxplot(x=[lm1[i, :, 3], lm7[i, :, 3]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 5], widths=0.7,
                boxprops=dict(facecolor=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Linear - FPR')
plt.legend(handles=[c1, c2, c3])

# logi
color = ['#00AF91', '#5089C6']

plt.subplot(322)
for i in range(logi1.shape[0]):
    plt.boxplot(x=[logi1[i, :, 0], logi7[i, :, 0]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 4], widths=0.7,
                boxprops=dict(facecolor=color[i]))

plt.xlabel('low corr                                         high corr')
plt.title('Logistic - AUC')

plt.subplot(324)
for i in range(logi1.shape[0]):
    plt.boxplot(x=[logi1[i, :, 1], logi7[i, :, 1]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 4], widths=0.7,
                boxprops=dict(facecolor=color[i]),
                medianprops=dict(color=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Logistic - Coefficient error')

plt.subplot(326)
for i in range(logi1.shape[0]):
    plt.boxplot(x=[logi1[i, :, 3], logi7[i, :, 3]], patch_artist=True,
                labels=['', ''], positions=[i + 1, i + 4], widths=0.7,
                boxprops=dict(facecolor=color[i]))
plt.xlabel('low corr                                         high corr')
plt.title('Logistic - FPR')
plt.legend(handles=[c1, c3])

plt.savefig('./perform.png')
print('Figure saved.')

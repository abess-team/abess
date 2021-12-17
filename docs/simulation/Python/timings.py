import os
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

os.system('/bin/python linear_test.py 0.1')
os.system('/bin/python linear_test.py 0.7')
os.system('/bin/python logi_test.py 0.1')
os.system('/bin/python logi_test.py 0.7')

# simulation results
lm1 = np.load('./Lm0.1_res.npy')
lm7 = np.load('./Lm0.7_res.npy')
logi1 = np.load('./Logistic0.1_res.npy')
logi7 = np.load('./Logistic0.7_res.npy')

plt.figure(figsize=(14, 6))

# lm_time
plt.subplot(121)
color = ['#00AF91', '#FFCC29', '#5089C6']
c1 = mpatches.Patch(color=color[0], label='Lasso')
c2 = mpatches.Patch(color=color[1], label='OMP')
c3 = mpatches.Patch(color=color[2], label='ABESS')

temp = np.vstack((lm1[:, [5, 11]], lm7[:, [5, 11]]))
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

temp = np.vstack((logi1[:, [5, 11]], logi7[:, [5, 11]]))
lm_time = plt.bar(x=[1, 2, 4, 5], height=temp[:, 0], yerr=temp[:, 1] * 2,
                  capsize=10, tick_label='', color=color[0:2])
plt.title('Logistic')
plt.ylabel('time(s)')
plt.ylim((0, 8))
plt.legend(handles=[c1, c3])
plt.xlabel('low corr                                                high corr')
# plt.savefig('./logi_time.png')

plt.savefig('./timings.png')
print('Figure saved.')

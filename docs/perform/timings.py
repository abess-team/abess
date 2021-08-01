import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

run = ''
while run != 'y' and run != 'n':
    run = input('Re-run the simulation? (y/n) ')
    if run == 'y':
        os.system('python linear_test.py 0.1')
        os.system('python linear_test.py 0.7')
        os.system('python logi_test.py 0.1')
        os.system('python logi_test.py 0.7')
        break
    
# simulation results
lm1 = pd.read_csv('./Lm0.1_res.csv', header=None)
lm7 = pd.read_csv('./Lm0.7_res.csv', header=None)
logi1 = pd.read_csv('./Logistic0.1_res.csv', header=None)
logi7 = pd.read_csv('./Logistic0.7_res.csv', header=None)

# bar plot setting
x = [1, 2, 4, 5]
label = ['abess-low','sklearn-low', 'abess-high', 'sklearn-high']
color = ['#00AF91', '#FFCC29', '#00AF91', '#FFCC29']
plt.figure(figsize = (14, 6))
# lm_time
plt.subplot(121)
temp = np.vstack((np.array(lm1)[:, [4, 9]], np.array(lm7)[:, [4, 9]]))
lm_time = plt.bar(x = x, height = temp[:, 0], yerr = temp[:, 1] * 2, capsize = 10, tick_label = label, color = color)
plt.title('Linear')
plt.ylabel('time(s)')
# plt.savefig('./lm_time.png')

# logi_time
plt.subplot(122)
temp = np.vstack((np.array(logi1)[:, [4, 9]], np.array(logi7)[:, [4, 9]]))
lm_time = plt.bar(x = x, height = temp[:, 0], yerr = temp[:, 1] * 2, capsize = 10, tick_label = label, color = color)
plt.title('Logistic')
plt.ylabel('time(s)')
# plt.savefig('./logi_time.png')

plt.savefig('./timings.png')
print('Figure saved.')
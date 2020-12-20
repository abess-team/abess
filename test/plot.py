import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib


plt.figure(figsize=(25,5), dpi=50)

#fig,axes = plt.subplots(4,5,figsize=(20,8))
#print(type(fig), '\n', type(axes))

data = pd.read_csv("simulate_lm_10_100.csv")

x_label = [200,400,600,800,1000,1200,1400,1600,1800,2000]

plt.subplot(1,4,1)  
plt.plot(x_label,data["TP_pdas_list"].values / 20, linestyle = '-',marker = '.', label='pdas')
plt.plot(x_label,data["TP_gpdas_list"].values / 20, linestyle = '-',marker = '.', label='pdas_gs')
plt.plot(x_label,data["TP_glmnet_list"].values / 20, linestyle = '-',marker = '.', label='lasso')
plt.xlabel("p")
plt.xticks(x_label)
plt.ylabel("TPR")
plt.yticks(np.linspace(0,1,5))

plt.subplot(1,4,2)  
plt.plot(x_label,data["FP_pdas_list"].values / (np.array(x_label) - 20))
plt.plot(x_label,data["FP_gpdas_list"].values / (np.array(x_label) - 20))
plt.plot(x_label,data["FP_glmnet_list"].values / (np.array(x_label) - 20))
plt.xlabel("p")
plt.xticks(x_label)
plt.ylabel("FPR")
plt.yticks(np.linspace(0,1,5))

plt.subplot(1,4,3)  
plt.plot(x_label,data["mcc_pdas_list"].values)
plt.plot(x_label,data["mcc_gpdas_list"].values)
plt.plot(x_label,data["mcc_glmnet_list"].values)
plt.xlabel("p")
plt.xticks(x_label)
plt.ylabel("mcc")
plt.yticks(np.linspace(0,1,5))

plt.subplot(1,4,4)  
plt.plot(x_label,data["pre_loss_pdas_list"].values, linestyle = '-',marker = '.', label='pdas')
plt.plot(x_label,data["pre_loss_gpdas_list"].values, linestyle = '-',marker = '.', label='pdas_gs')
plt.plot(x_label,data["pre_loss_glmnet_list"].values, linestyle = '-',marker = '.', label='lasso')
plt.xlabel("p")
plt.xticks(x_label)
plt.ylabel("mse")

#plt.plot(x_label,data["t_pdas_list"].values, linestyle = '-',marker = '.', label='pdas')
#plt.plot(x_label,data["t_gpdas_list"].values, linestyle = '-',marker = '.', label='pdas_gs')
#plt.plot(x_label,data["t_glmnet_list"].values, linestyle = '-',marker = '.', label='lasso')
#plt.xlabel("p")
#plt.xticks(x_label)
#plt.ylabel("t")



#plt.subplot(4,4,5)
#plt.subplot(4,4,6)
#plt.subplot(4,4,7)
#plt.subplot(4,4,8)
#plt.subplot(4,4,9)
#plt.subplot(4,4,10)
#plt.subplot(4,4,11)
#plt.subplot(4,4,12)
#plt.subplot(4,4,13)
#plt.subplot(4,4,14)
#plt.subplot(4,4,15)
#plt.subplot(4,4,16)
#plt.subplot(4,5,17)
#plt.subplot(4,5,18)
#plt.subplot(4,5,19)
#plt.subplot(4,5,20)


plt.legend()
plt.savefig("simulate_cox_10_10.png")
plt.show()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib


plt.figure(figsize=(20,12), dpi=400, facecolor='snow')

#fig,axes = plt.subplots(4,5,figsize=(20,8))
#print(type(fig), '\n', type(axes))

data = []
data.append(pd.read_csv("simulate_gaussian_10_100_final.csv"))
data.append(pd.read_csv("simulate_binomial_10_100_final.csv"))
#data.append(pd.read_csv("simulate_cox_10_100.csv"))
data.append(pd.read_csv("simulate_poisson_10_100_final.csv"))


x_label = [200,400,600,800,1000,1200,1400,1600,1800,2000]

label = ["LASSO", 'PDAS-SEQ-EBIC', 'PDAS-GS-EBIC', "PDAS-SEQ_CV", "PDAS-GS-CV"]
linewidth = 2
markersize = 10

mse_y_label = [8, 8, 6]
mse_e = [4, 1, 2]

nrow = 3
ncol = 3

for i in range(nrow):
    
    if i==0:
        plt.subplot(nrow,3,i + 1)  
        plt.plot(x_label,data[i]["TP_lasso_list"].values / 40, linestyle = '-',marker = '.', label='LASSO', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_pdas_list"].values / 40, linestyle = '-',marker = '.', label='PDAS_SEQ_EBIC', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_gpdas_list"].values / 40, linestyle = '-',marker = '.', label='PDAS_GS_EBIC', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_pdas_cv_list"].values / 20, linestyle = '-',marker = '.', label='PDAS_SEQ_CV', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_gpdas_cv_list"].values / 20, linestyle = '-',marker = '.', label='PDAS_GS_CV', linewidth=linewidth, markersize=markersize)
#        if i == nrow-1:
#            plt.xlabel("p")
        plt.xticks(x_label)
        plt.ylabel("TPR")
        plt.yticks(np.linspace(0.5,1,5))
        
        plt.subplot(nrow,3,i + 4)
        plt.plot(x_label,1 - data[i]["FP_lasso_list"].values / (np.array(x_label) - 40), linestyle = '-',marker = '.', label='lasso', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_pdas_list"].values / (np.array(x_label) - 40), linestyle = '-',marker = '.', label='pdas', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_gpdas_list"].values / (np.array(x_label) - 40), linestyle = '-',marker = '.', label='pdas_gs', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_pdas_cv_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='pdas_cv', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_gpdas_cv_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='pdas_gs_cv', linewidth=linewidth, markersize=markersize)
#        plt.xlabel("p")
        plt.xticks(x_label)
        plt.ylabel("TNR")
        plt.yticks(np.linspace(0.5,1,5))
    else:
        plt.subplot(nrow,3,i + 1)
        plt.plot(x_label,data[i]["TP_lasso_list"].values / 20, linestyle = '-',marker = '.', label=label[0], linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_pdas_list"].values / 20, linestyle = '-',marker = '.', label=label[1], linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_gpdas_list"].values / 20, linestyle = '-',marker = '.', label=label[2], linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_pdas_cv_list"].values / 20, linestyle = '-',marker = '.', label=label[3], linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,data[i]["TP_gpdas_cv_list"].values / 20, linestyle = '-',marker = '.', label=label[4], linewidth=linewidth, markersize=markersize)
#        if i == nrow-1:
#            plt.xlabel("p")
        plt.xticks(x_label)
        plt.ylabel("TPR")
        plt.yticks(np.linspace(0.5,1,5))
        
        plt.subplot(nrow,3,i + 4)
        plt.plot(x_label,1 - data[i]["FP_lasso_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='lasso', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_pdas_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='pdas', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_gpdas_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='pdas_gs', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_pdas_cv_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='pdas_cv', linewidth=linewidth, markersize=markersize)
        plt.plot(x_label,1 - data[i]["FP_gpdas_cv_list"].values / (np.array(x_label) - 20), linestyle = '-',marker = '.', label='pdas_gs_cv', linewidth=linewidth, markersize=markersize)
#        if i == nrow-1:
#            plt.xlabel("p")
        plt.xticks(x_label)
        plt.ylabel("TNR")
        plt.yticks(np.linspace(0.5,1,5))
    ax = plt.subplot(nrow,3,i + 7)  
    plt.plot(x_label,data[i]["mcc_lasso_list"].values, linestyle = '-',marker = '.', label=label[0], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["mcc_pdas_list"].values, linestyle = '-',marker = '.',label=label[1], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["mcc_gpdas_list"].values, linestyle = '-',marker = '.', label=label[2], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["mcc_pdas_cv_list"].values, linestyle = '-',marker = '.',label=label[3], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["mcc_gpdas_cv_list"].values, linestyle = '-',marker = '.', label=label[4], linewidth=linewidth, markersize=markersize)
#    if i == nrow-1:
#        plt.xlabel("p")
    plt.xticks(x_label)
    plt.xlabel("p")
    plt.ylabel("MCC")
#    yticks = ["%.3f" %y for y in [0.000, 0.250, 0.500, 0.750, 1.000]]
    plt.yticks([0.000, 0.250, 0.500, 0.750, 1.000])
    ax.set_yticklabels("%.3f" %y for y in [0.000, 0.250, 0.500, 0.750, 1.000])
    
    
plt.legend(loc=2, bbox_to_anchor=(-1.55,3.6),borderaxespad = 0.,ncol=5, fancybox=True, markerscale=1.6)  
plt.savefig("simulate_all.pdf", transparent=True)
#plt.show()




plt.figure(figsize=(20,4), dpi=400)

#fig,axes = plt.subplots(4,5,figsize=(20,8))
#print(type(fig), '\n', type(axes))

#data = []
#data.append(pd.read_csv("simulate_gaussian_10_100.csv"))
#data.append(pd.read_csv("simulate_binomial_10_100.csv"))
#data.append(pd.read_csv("simulate_cox_10_100.csv"))
#data.append(pd.read_csv("simulate_poisson_10_100.csv"))
#
#
#x_label = [200,400,600,800,1000,1200,1400,1600,1800,2000]

for i in range(nrow):
    ax = plt.subplot(1, nrow, i*1 + 1)
    plt.plot(x_label,data[i]["pre_loss_lasso_list"].values, linestyle = '-',marker = '.', label=label[0], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["pre_loss_pdas_list"].values, linestyle = '-',marker = '.', label=label[1], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["pre_loss_gpdas_list"].values, linestyle = '-',marker = '.', label=label[2], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["pre_loss_pdas_cv_list"].values, linestyle = '-',marker = '.', label=label[3], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["pre_loss_gpdas_cv_list"].values, linestyle = '-',marker = '.', label=label[4], linewidth=linewidth, markersize=markersize)
    plt.xlabel("p")
    plt.xticks(x_label)
    plt.ylabel("RMSE (x e-" + str(mse_e[i]) + ')')
    plt.yticks(np.linspace(0, mse_y_label[i] * 10 ** (-mse_e[i]), 5))
    ax.set_yticklabels("%.1f" % y for y in np.linspace(0, mse_y_label[i], 5))
    
plt.legend(loc=2, bbox_to_anchor=(-1.55,1.15),borderaxespad = 0.,ncol=5, fancybox=True, markerscale=1.6)
plt.savefig("simulate_all_pre.pdf", transparent=True)
#plt.show()



plt.figure(figsize=(20,4), dpi=400)

#fig,axes = plt.subplots(4,5,figsize=(20,8))
#print(type(fig), '\n', type(axes))

#data = []
#data.append(pd.read_csv("simulate_gaussian_10_100.csv"))
#data.append(pd.read_csv("simulate_binomial_10_100.csv"))
#data.append(pd.read_csv("simulate_cox_10_100.csv"))
#data.append(pd.read_csv("simulate_poisson_10_100.csv"))
#
#
#x_label = [200,400,600,800,1000,1200,1400,1600,1800,2000]

for i in range(nrow):
    plt.subplot(1, nrow, i*1 + 1)
    plt.plot(x_label,data[i]["t_lasso_list"].values, linestyle = '-',marker = '.', label=label[0], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["t_pdas_list"].values, linestyle = '-',marker = '.', label=label[1], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["t_gpdas_list"].values, linestyle = '-',marker = '.', label=label[2], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["t_pdas_cv_list"].values, linestyle = '-',marker = '.', label=label[3], linewidth=linewidth, markersize=markersize)
    plt.plot(x_label,data[i]["t_gpdas_cv_list"].values, linestyle = '-',marker = '.', label=label[4], linewidth=linewidth, markersize=markersize)
    plt.xlabel("p")
    plt.xticks(x_label)
    plt.ylabel("Run Time (s)")
plt.legend(loc=2, bbox_to_anchor=(-1.55,1.15),borderaxespad = 0.,ncol=5, fancybox=True, markerscale=1.6)

#ax = plt.subplot(1, nrow, 1)
#ax.set_yticklabels("%.1e" % y for y in np.linspace(0, 0.0006, 5))
#
#ax = plt.subplot(1, nrow, 2)
#ax.set_yticklabels("%.1e" % y for y in np.linspace(0, 0.8, 5))
#
#ax = plt.subplot(1, nrow, 3)
#ax.set_yticklabels("%.1e" % y for y in np.linspace(0, 0.05, 5))


plt.savefig("simulate_all_t.pdf", transparent=True)
#plt.show()











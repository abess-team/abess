setwd('C:/Users/mac/OneDrive/coding/frame/l0l2_frame/l0l2_paper/simulation_bess')
library(glmnet)
library(mccr)
library(ROCR)
library(MASS)
library(BeSS)
library(ElemStatLearn)
generatedata=function(x,b,b_0){
  a=exp(b_0+t(x)%*%b)/(1+exp(b_0+t(x)%*%b))
  if(is.infinite(exp(b_0+t(x)%*%b))){
    a=1
  }
  return(a)
}
mysd=function(y) {
  if(sqrt(sum((y-mean(y))^2)/length(y))==0){
    return(1)
  }else{
    return(sqrt(sum((y-mean(y))^2)/length(y))) 
  }
}
#####################################################################################
#######################实验一######################################################
testtms=100
n=200
p=50
n_train=100
beta=rep(c(1,rep(0,4)),10)
k=length(which(beta!=0))
subset=ifelse(beta==0,0,1)

sigma=0.6*diag(p)+0.4

# sigma <- matrix(0, nrow = p, ncol = p)
# for (i in 1:p){
#   for (j in 1:p){
#     sigma[i, j] <- 0.5^abs(i-j)
#   }
# }
SNRlist=seq(0.01,100,length.out = 10)
snrmax=100
snrmin=0.01
lnmax=log(snrmax)
lnmin=log(snrmin)
lnseq=seq(lnmin,lnmax,length.out = 10)
SNRlist=exp(lnseq)
namelist=character()
for(k in 1:10){
  namelist[k]=paste('SNR',k,sep='')
}

pe_l0l2_low=rep(0,testtms)
tpr_l0l2_low=rep(0,testtms)
fpr_l0l2_low=rep(0,testtms)
norm_l0l2_low=rep(0,testtms)
t_l0l2_low=rep(0,testtms)
mcc_l0l2_low=rep(0,testtms)
nz_l0l2_low=rep(0,testtms)

pe_ela_low=rep(0,testtms)
tpr_ela_low=rep(0,testtms)
fpr_ela_low=rep(0,testtms)
norm_ela_low=rep(0,testtms)
t_ela_low=rep(0,testtms)
mcc_ela_low=rep(0,testtms)
nz_ela_low=rep(0,testtms)

pe_lasso_low=rep(0,testtms)
tpr_lasso_low=rep(0,testtms)
fpr_lasso_low=rep(0,testtms)
norm_lasso_low=rep(0,testtms)
t_lasso_low=rep(0,testtms)
mcc_lasso_low=rep(0,testtms)
nz_lasso_low=rep(0,testtms)

pe_pdas_low=rep(0,testtms)
tpr_pdas_low=rep(0,testtms)
fpr_pdas_low=rep(0,testtms)
norm_pdas_low=rep(0,testtms)
t_pdas_low=rep(0,testtms)
mcc_pdas_low=rep(0,testtms)
nz_pdas_low=rep(0,testtms)

pe_ridge_low=rep(0,testtms)
tpr_ridge_low=rep(0,testtms)
fpr_ridge_low=rep(0,testtms)
norm_ridge_low=rep(0,testtms)
t_ridge_low=rep(0,testtms)
mcc_ridge_low=rep(0,testtms)
nz_ridge_low=rep(0,testtms)

beta_l0l2_low=numeric()
beta_lasso_low=numeric()
beta_ela_low=numeric()
beta_pdas_low=numeric()
beta_ridge_low=numeric()
for(j in 1:1){
  sigma2=sqrt((t(beta)%*%sigma%*%beta)/SNRlist[j])
  
  ######开始#####
  for(i in 1:testtms){
    print(paste("第",i))
    set.seed(i)
    X=mvrnorm(n,rep(0,p),sigma);
    set.seed(i+2)
    y=X%*%beta+rnorm(n,0,sigma2);
    X_train=X[1:n_train,]
    y_train=y[1:n_train]
    X_test=X[(n_train+1):n,]
    y_test=y[(n_train+1):n]
    
    ridge.mod=glmnet(X_train,y_train,alpha=0)
    cv.out_ridge=cv.glmnet(X_train,y_train,alpha=0)
    lambda_2=cv.out_ridge$lambda.min
    #lammax=ifelse(lambda_2>10000,0.001*lambda_2,0.1*lambda_2)
    t1=proc.time()
    list_l0l2=bessCpp(X_train,y_train,1,rep(1,n_train),T,1,1,20,10,1,T,1,T,5,rep(2,10),1:20,seq(0.001*lambda_2,0.000001*lambda_2,length.out = 100),10,10,10,10,F)
    t2=proc.time()
    t_l0l2_low[i]=(t2-t1)[3][[1]]
    
    t1=proc.time()
    list_pdas=bessCpp(X_train,y_train,1,rep(1,n_train),T,1,1,20,10,1,T,1,T,5,rep(2,10),1:20,0,10,10,10,10,F)
    t2=proc.time()
    t_pdas_low[i]=(t2-t1)[3][[1]]
    
    t1=proc.time()
    cvob_lasso=cv.glmnet(X_train,y_train)
    beta_lasso_temp=predict(cvob_lasso,type='coefficients',s=cvob_lasso$lambda.min)
    t2=proc.time()
    t_lasso_low[i]=(t2-t1)[3][[1]]
    
    t1=proc.time()
    cvob_ela=cv.glmnet(X_train,y_train,alpha=0.5)
    beta_ela_temp=predict(cvob_ela,type='coefficients',s=cvob_ela$lambda.min)
    t2=proc.time()
    t_lasso_low[i]=(t2-t1)[3][[1]]
    
    ##############
    pe_l0l2_low[i]=norm(y_test-(X_test%*%list_l0l2$beta+list_l0l2$coef0),type = '2')^2/norm(y_test,type='2')^2
    norm_l0l2_low[i]=norm(c(list_l0l2$coef0,list_l0l2$beta-beta),type='2')
    mcc_l0l2_low[i]=mccr(subset,ifelse(list_l0l2$beta==0,0,1))
    tpr_l0l2_low[i]=sum(ifelse(beta!=0&list_l0l2$beta!=0,1,0))/k
    fpr_l0l2_low[i]=sum(ifelse(beta==0&list_l0l2$beta!=0,1,0))/(p-k)
    nz_l0l2_low[i]=length(which(list_l0l2$beta!=0))/k
    
    pe_pdas_low[i]=norm(y_test-(X_test%*%list_pdas$beta+list_pdas$coef0),type = '2')^2/norm(y_test,type='2')^2
    norm_pdas_low[i]=norm(c(list_pdas$coef0,list_pdas$beta-beta),type='2')
    mcc_pdas_low[i]=mccr(subset,ifelse(list_pdas$beta==0,0,1))
    tpr_pdas_low[i]=sum(ifelse(beta!=0&list_pdas$beta!=0,1,0))/k
    fpr_pdas_low[i]=sum(ifelse(beta==0&list_pdas$beta!=0,1,0))/(p-k)
    nz_pdas_low[i]=length(which(list_pdas$beta!=0))/k
    
    pe_ela_low[i]=norm(y_test-predict(cvob_ela,s=cvob_ela$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
    norm_ela_low[i]=norm(c(beta_ela_temp[1],beta_ela_temp[2:(p+1)]-beta),type='2')
    mcc_ela_low[i]=mccr(subset,ifelse(beta_ela_temp[2:(p+1)]==0,0,1))
    tpr_ela_low[i]=sum(ifelse(beta!=0&beta_ela_temp[2:(p+1)]!=0,1,0))/k
    fpr_ela_low[i]=sum(ifelse(beta==0&beta_ela_temp[2:(p+1)]!=0,1,0))/(p-k)
    nz_ela_low[i]=length(which(beta_ela_temp[2:(p+1)]!=0))/k
    
    pe_lasso_low[i]=norm(y_test-predict(cvob_lasso,s=cvob_lasso$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
    norm_lasso_low[i]=norm(c(beta_lasso_temp[1],beta_lasso_temp[2:(p+1)]-beta),type='2')
    mcc_lasso_low[i]=mccr(subset,ifelse(beta_lasso_temp[2:(p+1)]==0,0,1))
    tpr_lasso_low[i]=sum(ifelse(beta!=0&beta_lasso_temp[2:(p+1)]!=0,1,0))/k
    fpr_lasso_low[i]=sum(ifelse(beta==0&beta_lasso_temp[2:(p+1)]!=0,1,0))/(p-k)
    nz_lasso_low[i]=length(which(beta_lasso_temp[2:(p+1)]!=0))/k
    
    
  }
  mean_pe_l0l2_low=mean(pe_l0l2_low)
  mean_tpr_l0l2_low=mean(tpr_l0l2_low)
  mean_fpr_l0l2_low=mean(fpr_l0l2_low)
  mean_t_l0l2_low=mean(t_l0l2_low)
  mean_norm_l0l2_low=mean(norm_l0l2_low)
  mean_nz_l0l2_low=mean(nz_l0l2_low)
  mean_mcc_l0l2_low=mean(mcc_l0l2_low)
  
  mean_pe_pdas_low=mean(pe_pdas_low)
  mean_tpr_pdas_low=mean(tpr_pdas_low)
  mean_fpr_pdas_low=mean(fpr_pdas_low)
  mean_t_pdas_low=mean(t_pdas_low)
  mean_norm_pdas_low=mean(norm_pdas_low)
  mean_nz_pdas_low=mean(nz_pdas_low)
  mean_mcc_pdas_low=mean(mcc_pdas_low)
  
  mean_pe_lasso_low=mean(pe_lasso_low)
  mean_tpr_lasso_low=mean(tpr_lasso_low)
  mean_fpr_lasso_low=mean(fpr_lasso_low)
  mean_t_lasso_low=mean(t_lasso_low)
  mean_norm_lasso_low=mean(norm_lasso_low)
  mean_nz_lasso_low=mean(nz_lasso_low)
  mean_mcc_lasso_low=mean(mcc_lasso_low)
  
  mean_pe_ela_low=mean(pe_ela_low)
  mean_tpr_ela_low=mean(tpr_ela_low)
  mean_fpr_ela_low=mean(fpr_ela_low)
  mean_t_ela_low=mean(t_ela_low)
  mean_norm_ela_low=mean(norm_ela_low)
  mean_nz_ela_low=mean(nz_ela_low)
  mean_mcc_ela_low=mean(mcc_ela_low)
  
  data1=data.frame(mean_pe_lasso_low, mean_pe_l0l2_low,mean_pe_ela_low,mean_pe_pdas_low,
                   mean_tpr_lasso_low,mean_tpr_l0l2_low,mean_tpr_ela_low,mean_tpr_pdas_low,
                   mean_fpr_lasso_low,mean_fpr_l0l2_low,mean_fpr_ela_low,mean_fpr_pdas_low,
                   mean_t_lasso_low,mean_t_l0l2_low,mean_t_ela_low,mean_t_pdas_low,
                   mean_norm_lasso_low,mean_norm_l0l2_low,mean_norm_ela_low,mean_norm_pdas_low,
                   mean_nz_lasso_low,mean_nz_l0l2_low,mean_nz_ela_low,mean_nz_pdas_low,
                   mean_mcc_lasso_low,mean_mcc_l0l2_low,mean_mcc_ela_low,mean_mcc_pdas_low)
  SNRdata=rbind(SNRdata,data1)
}
save(SNRdata,file = 'SNR p=50.RData')
####################################ridge#################################################
for(j in 2:10){
  sigma2=sqrt((t(beta)%*%sigma%*%beta)/SNRlist[j])
  
  ######开始#####
  for(i in 1:testtms){
    print(paste("第",i))
    set.seed(i)
    X=mvrnorm(n,rep(0,p),sigma);
    set.seed(i+2)
    y=X%*%beta+rnorm(n,0,sigma2);
    X_train=X[1:n_train,]
    y_train=y[1:n_train]
    X_test=X[(n_train+1):n,]
    y_test=y[(n_train+1):n]
    
    t1=proc.time()
    cvob_ridge=cv.glmnet(X_train,y_train,alpha=0)
    beta_ridge_temp=predict(cvob_ela,type='coefficients',s=cvob_ela$lambda.min)
    t2=proc.time()
    t_ridge_low[i]=(t2-t1)[3][[1]]
    
    
    
    ##############
    pe_ridge_low[i]=norm(y_test-predict(cvob_ridge,s=cvob_ridge$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
    norm_ridge_low[i]=norm(c(beta_ridge_temp[1],beta_ridge_temp[2:(p+1)]-beta),type='2')
    mcc_ridge_low[i]=mccr(subset,ifelse(beta_ridge_temp[2:(p+1)]==0,0,1))
    tpr_ridge_low[i]=sum(ifelse(beta!=0&beta_ridge_temp[2:(p+1)]!=0,1,0))/k
    fpr_ridge_low[i]=sum(ifelse(beta==0&beta_ridge_temp[2:(p+1)]!=0,1,0))/(p-k)
    nz_ridge_low[i]=length(which(beta_ridge_temp[2:(p+1)]!=0))/k
    
    
  }
  
  
  mean_pe_ridge_low=mean(pe_ridge_low)
  mean_tpr_ridge_low=mean(tpr_ridge_low)
  mean_fpr_ridge_low=mean(fpr_ridge_low)
  mean_t_ridge_low=mean(t_ridge_low)
  mean_norm_ridge_low=mean(norm_ridge_low)
  mean_nz_ridge_low=mean(nz_ridge_low)
  mean_mcc_ridge_low=mean(mcc_ridge_low)
  
  data2=data.frame(mean_pe_ridge_low,
                   mean_tpr_ridge_low,
                   mean_fpr_ridge_low,
                   mean_t_ridge_low,
                   mean_norm_ridge_low,
                   mean_nz_ridge_low,
                   mean_mcc_ridge_low
  )
  riddata=rbind(riddata,data2)
}
#######################实验二######################################################
#############################
testtms=100
n=2000
p=1000
n_train=1000
beta=rep(c(1,rep(0,49)),20)
k=length(which(beta!=0))
subset=ifelse(beta==0,0,1)

sigma=0.6*diag(p)+0.4

# sigma <- matrix(0, nrow = p, ncol = p)
# for (i in 1:p){
#   for (j in 1:p){
#     sigma[i, j] <- 0.5^abs(i-j)
#   }
# }
#SNRlist=seq(0.01,100,length.out = 10)
snrmax=100
snrmin=0.01
lnmax=log(snrmax)
lnmin=log(snrmin)
lnseq=seq(lnmin,lnmax,length.out = 10)
SNRlist=exp(lnseq)
namelist=character()
for(k in 1:10){
  namelist[k]=paste('SNR',k,sep='')
}

pe_l0l2_low=rep(0,testtms)
tpr_l0l2_low=rep(0,testtms)
fpr_l0l2_low=rep(0,testtms)
norm_l0l2_low=rep(0,testtms)
t_l0l2_low=rep(0,testtms)
mcc_l0l2_low=rep(0,testtms)
nz_l0l2_low=rep(0,testtms)

pe_ela_low=rep(0,testtms)
tpr_ela_low=rep(0,testtms)
fpr_ela_low=rep(0,testtms)
norm_ela_low=rep(0,testtms)
t_ela_low=rep(0,testtms)
mcc_ela_low=rep(0,testtms)
nz_ela_low=rep(0,testtms)

pe_lasso_low=rep(0,testtms)
tpr_lasso_low=rep(0,testtms)
fpr_lasso_low=rep(0,testtms)
norm_lasso_low=rep(0,testtms)
t_lasso_low=rep(0,testtms)
mcc_lasso_low=rep(0,testtms)
nz_lasso_low=rep(0,testtms)

pe_pdas_low=rep(0,testtms)
tpr_pdas_low=rep(0,testtms)
fpr_pdas_low=rep(0,testtms)
norm_pdas_low=rep(0,testtms)
t_pdas_low=rep(0,testtms)
mcc_pdas_low=rep(0,testtms)
nz_pdas_low=rep(0,testtms)

beta_l0l2_low=numeric()
beta_lasso_low=numeric()
beta_ela_low=numeric()
beta_pdas_low=numeric()
lambdaseq=c(12027.82,7422.049,2101.271,824.1841,305.489,108.1268,89.72172,89.69352,89.67661, 89.66648)
for(j in 6:10){
  sigma2=sqrt((t(beta)%*%sigma%*%beta)/SNRlist[j])
  
  ######开始#####
  for(i in 1:testtms){
    print(paste("第",i))
    set.seed(i)
    X=mvrnorm(n,rep(0,p),sigma);
    set.seed(i+2)
    y=X%*%beta+rnorm(n,0,sigma2);
    X_train=X[1:n_train,]
    y_train=y[1:n_train]
    X_test=X[(n_train+1):n,]
    y_test=y[(n_train+1):n]
    
#     ridge.mod=glmnet(X_train,y_train,alpha=0)
#     cv.out_ridge=cv.glmnet(X_train,y_train,alpha=0)
#     lambda_2=cv.out_ridge$lambda.min
# lambda_2
#     lammax=ifelse(lambdaseq[j]>100,0.01*lambdaseq[j],lambdaseq[j])
    if(lambdaseq[j]>10000){
      lammax=lambdaseq[j]*0.001
    }else if(lambdaseq[j]>1000&lambdaseq[j]<=10000){
      lammax=lambdaseq[j]*0.01
    }else{
      lammax=lambdaseq[j]*0.1
    }
    
    #lammax=ifelse(lambdaseq[j]<100,0.1*lambdaseq[j],0.01*lambdaseq[j])
    t1=proc.time()
    list_l0l2=bessCpp(X_train,y_train,1,rep(1,n_train),T,1,1,20,10,1,T,1,T,5,rep(2,10),1:20,seq(lammax,0.001*lammax,length.out = 100),10,10,10,10,T)
    t2=proc.time()
    t_l0l2_low[i]=(t2-t1)[3][[1]]
    
    t1=proc.time()
    list_pdas=bessCpp(X_train,y_train,1,rep(1,n_train),T,1,1,20,10,1,T,1,T,5,rep(2,10),1:20,0,10,10,10,10,T)
    t2=proc.time()
    t_pdas_low[i]=(t2-t1)[3][[1]]
    
    t1=proc.time()
    cvob_lasso=cv.glmnet(X_train,y_train)
    beta_lasso_temp=predict(cvob_lasso,type='coefficients',s=cvob_lasso$lambda.min)
    t2=proc.time()
    t_lasso_low[i]=(t2-t1)[3][[1]]
    
    t1=proc.time()
    cvob_ela=cv.glmnet(X_train,y_train,alpha=0.5)
    beta_ela_temp=predict(cvob_ela,type='coefficients',s=cvob_ela$lambda.min)
    t2=proc.time()
    t_lasso_low[i]=(t2-t1)[3][[1]]
    
    ##############
    pe_l0l2_low[i]=norm(y_test-(X_test%*%list_l0l2$beta+list_l0l2$coef0),type = '2')^2/norm(y_test,type='2')^2
    norm_l0l2_low[i]=norm(c(list_l0l2$coef0,list_l0l2$beta-beta),type='2')
    mcc_l0l2_low[i]=mccr(subset,ifelse(list_l0l2$beta==0,0,1))
    tpr_l0l2_low[i]=sum(ifelse(beta!=0&list_l0l2$beta!=0,1,0))/k
    fpr_l0l2_low[i]=sum(ifelse(beta==0&list_l0l2$beta!=0,1,0))/(p-k)
    nz_l0l2_low[i]=length(which(list_l0l2$beta!=0))/k
    
    pe_pdas_low[i]=norm(y_test-(X_test%*%list_pdas$beta+list_pdas$coef0),type = '2')^2/norm(y_test,type='2')^2
    norm_pdas_low[i]=norm(c(list_pdas$coef0,list_pdas$beta-beta),type='2')
    mcc_pdas_low[i]=mccr(subset,ifelse(list_pdas$beta==0,0,1))
    tpr_pdas_low[i]=sum(ifelse(beta!=0&list_pdas$beta!=0,1,0))/k
    fpr_pdas_low[i]=sum(ifelse(beta==0&list_pdas$beta!=0,1,0))/(p-k)
    nz_pdas_low[i]=length(which(list_pdas$beta!=0))/k
    
    pe_ela_low[i]=norm(y_test-predict(cvob_ela,s=cvob_ela$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
    norm_ela_low[i]=norm(c(beta_ela_temp[1],beta_ela_temp[2:(p+1)]-beta),type='2')
    mcc_ela_low[i]=mccr(subset,ifelse(beta_ela_temp[2:(p+1)]==0,0,1))
    tpr_ela_low[i]=sum(ifelse(beta!=0&beta_ela_temp[2:(p+1)]!=0,1,0))/k
    fpr_ela_low[i]=sum(ifelse(beta==0&beta_ela_temp[2:(p+1)]!=0,1,0))/(p-k)
    nz_ela_low[i]=length(which(beta_ela_temp[2:(p+1)]!=0))/k
    
    pe_lasso_low[i]=norm(y_test-predict(cvob_lasso,s=cvob_lasso$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
    norm_lasso_low[i]=norm(c(beta_lasso_temp[1],beta_lasso_temp[2:(p+1)]-beta),type='2')
    mcc_lasso_low[i]=mccr(subset,ifelse(beta_lasso_temp[2:(p+1)]==0,0,1))
    tpr_lasso_low[i]=sum(ifelse(beta!=0&beta_lasso_temp[2:(p+1)]!=0,1,0))/k
    fpr_lasso_low[i]=sum(ifelse(beta==0&beta_lasso_temp[2:(p+1)]!=0,1,0))/(p-k)
    nz_lasso_low[i]=length(which(beta_lasso_temp[2:(p+1)]!=0))/k
    
    
  }
  mean_pe_l0l2_low=mean(pe_l0l2_low)
  mean_tpr_l0l2_low=mean(tpr_l0l2_low)
  mean_fpr_l0l2_low=mean(fpr_l0l2_low)
  mean_t_l0l2_low=mean(t_l0l2_low)
  mean_norm_l0l2_low=mean(norm_l0l2_low)
  mean_nz_l0l2_low=mean(nz_l0l2_low)
  mean_mcc_l0l2_low=mean(mcc_l0l2_low)
  
  mean_pe_pdas_low=mean(pe_pdas_low)
  mean_tpr_pdas_low=mean(tpr_pdas_low)
  mean_fpr_pdas_low=mean(fpr_pdas_low)
  mean_t_pdas_low=mean(t_pdas_low)
  mean_norm_pdas_low=mean(norm_pdas_low)
  mean_nz_pdas_low=mean(nz_pdas_low)
  mean_mcc_pdas_low=mean(mcc_pdas_low)
  
  mean_pe_lasso_low=mean(pe_lasso_low)
  mean_tpr_lasso_low=mean(tpr_lasso_low)
  mean_fpr_lasso_low=mean(fpr_lasso_low)
  mean_t_lasso_low=mean(t_lasso_low)
  mean_norm_lasso_low=mean(norm_lasso_low)
  mean_nz_lasso_low=mean(nz_lasso_low)
  mean_mcc_lasso_low=mean(mcc_lasso_low)
  
  mean_pe_ela_low=mean(pe_ela_low)
  mean_tpr_ela_low=mean(tpr_ela_low)
  mean_fpr_ela_low=mean(fpr_ela_low)
  mean_t_ela_low=mean(t_ela_low)
  mean_norm_ela_low=mean(norm_ela_low)
  mean_nz_ela_low=mean(nz_ela_low)
  mean_mcc_ela_low=mean(mcc_ela_low)
  
  data1=data.frame(mean_pe_lasso_low, mean_pe_l0l2_low,mean_pe_ela_low,mean_pe_pdas_low,
                   mean_tpr_lasso_low,mean_tpr_l0l2_low,mean_tpr_ela_low,mean_tpr_pdas_low,
                   mean_fpr_lasso_low,mean_fpr_l0l2_low,mean_fpr_ela_low,mean_fpr_pdas_low,
                   mean_t_lasso_low,mean_t_l0l2_low,mean_t_ela_low,mean_t_pdas_low,
                   mean_norm_lasso_low,mean_norm_l0l2_low,mean_norm_ela_low,mean_norm_pdas_low,
                   mean_nz_lasso_low,mean_nz_l0l2_low,mean_nz_ela_low,mean_nz_pdas_low,
                   mean_mcc_lasso_low,mean_mcc_l0l2_low,mean_mcc_ela_low,mean_mcc_pdas_low)
  SNRdata=rbind(SNRdata,data1)
}
save(SNRdata,file = 'SNR p=1000 constant 1-10 new.RData')
SNRdata2=SNRdata
SNRdata2$mean_tpr_l0l2_low=SNRdata$mean_tpr_l0l2_low/2
SNRdata2$mean_fpr_l0l2_low=SNRdata$mean_fpr_l0l2_low*990/980
SNRdata2$mean_nz_l0l2_low=SNRdata$mean_nz_l0l2_low/2

SNRdata2$mean_tpr_pdas_low=SNRdata$mean_tpr_pdas_low/2
SNRdata2$mean_fpr_pdas_low=SNRdata$mean_fpr_pdas_low*990/980
SNRdata2$mean_nz_pdas_low=SNRdata$mean_nz_pdas_low/2

SNRdata2$mean_tpr_lasso_low=SNRdata$mean_tpr_lasso_low/2
SNRdata2$mean_fpr_lasso_low=SNRdata$mean_fpr_lasso_low*990/980
SNRdata2$mean_nz_lasso_low=SNRdata$mean_nz_lasso_low/2

SNRdata2$mean_tpr_ela_low=SNRdata$mean_tpr_ela_low/2
SNRdata2$mean_fpr_ela_low=SNRdata$mean_fpr_ela_low*990/980
SNRdata2$mean_nz_ela_low=SNRdata$mean_nz_ela_low/2
###################作图#########################
library(tidyr)# 使用的gather & spread
library(reshape2) # 使用的函数 melt & dcast 
library(cowplot)
library(ggthemes)
library(ggplot2)
p1=ggplot(SNRdata,aes(SNRlist,mean_pe_l0l2_low))
pe=SNRdata[,1:4]
tpr=SNRdata[,5:8]
fpr=SNRdata[,9:12]
t=SNRdata[,13:16]
norm=SNRdata[,17:20]
nz=k*SNRdata[,21:24]
mcc=SNRdata[,25:28]
Method=c(rep('Lasso',10),rep('L0L2',10),rep('Elastic Net',10),rep('PDAS',10))

pe$meric=rep('Prediction Error',10)
pe_fram=gather(pe,Method,value,1:4)
pe_fram$Method=Method

tpr$meric=rep('TPR',10)
tpr_fram=gather(tpr,Method,value,1:4)
tpr_fram$Method=Method

fpr$meric=rep('FPR',10)
fpr_fram=gather(fpr,Method,value,1:4)
fpr_fram$Method=Method

norm$meric=rep('Error Bound',10)
norm_fram=gather(norm,Method,value,1:4)
norm_fram$Method=Method
nz$meric=rep('Model Size',10)
nz_fram=gather(nz,Method,value,1:4)
nz_fram$Method=Method
mcc$meric=rep('MCC',10)
mcc_fram=gather(mcc,Method,value,1:4)
mcc_fram$Method=Method

data_long=rbind(pe_fram,tpr_fram,fpr_fram,norm_fram,nz_fram,mcc_fram)
data_long$SNR=rep(SNRlist,24)
scientific_10 <- function(x) {
  parse(text=gsub( "1 e +"," %*% 10^", scales::scientific_format()(x)))
}
p1=ggplot(data_long,aes(x=SNR,y=value,group=Method))+
  geom_line(aes(color=Method),size=1)+
  geom_point(aes(color=Method,shape=Method))+
  #scale_linetype_manual(values=c("twodash", "dotted"))+
  #theme_few(base_size = 14)+

  facet_wrap(~meric, scales="free_y", nrow = 2) +
  scale_x_log10(labels = scientific_10)+
  theme(axis.title.x=element_text(size = 13),
        # axis.text.x = element_text(angle = 30),
        axis.text.x = element_text(),
        axis.title.y=element_blank(),
        #axis.text.y=element_blank(),
        #legend.position = "none"
        legend.position = "bottom",
        legend.title=element_text(size=13),
        legend.text=element_text(size=13),
        #plot.title = element_text(size = 40, face = "bold"),
        strip.text = element_text(size=13)
  )
p1
ggsave(filename = 'LinearSimulation_high_exponential_new.eps',dpi=300,width=10.032,height = 7.464)


###############时间###########
lowcon=SNRdata
lowcont=lowcon[,13:16]
lowext=lowex[,13:16]
highext=data[1:10,13:16]
highcont=data[11:20,13:16]

Method=c(rep('Lasso',10),rep('L0L2',10),rep('Elastic Net',10),rep('PDAS',10))
lowcont$case=rep('Experiment 1',10)
lowcont_fram=gather(lowcont,Method,value,1:4)
lowcont_fram$Method=Method

lowext$case=rep('Experiment 2',10)
lowext_fram=gather(lowext,Method,value,1:4)
lowext_fram$Method=Method

highcont$case=rep('Experiment 3',10)
highcont_fram=gather(highcont,Method,value,1:4)
highcont_fram$Method=Method

highext$case=rep('Experiment 4',10)
highext_fram=gather(highext,Method,value,1:4)
highext_fram$Method=Method

data_long2=rbind(lowext_fram,lowcont_fram,highext_fram,highcont_fram)
data_long2$SNR=SNRlist
data_long2$case=factor(data_long2$case,levels = c('p=50, k=10, n=100, exponential correlation','p=50, k=10, n=100, contant correlation',
                       'p=1000, k=20, n=1000, exponential correlation','p=1000, k=20, n=1000, constant correlation'))
# #data_long2=within(data_long2,case=factor(case,levels = c('p=50, k=10, n=100, exponential correlation','p=50, k=10, n=100, contant correlation',
#                                                          'p=1000, k=20, n=1000, exponential correlation','p=1000, k=20, n=1000, constant correlation')))
label=c('p=50, k=10, n=100, exponential correlation'='Experiment 1',
        'p=50, k=10, n=100, contant correlation'='Experiment 2',
        'p=1000, k=20, n=1000, exponential correlation'='Experiment 3',
        'p=1000, k=20, n=1000, constant correlation'='Experiment 4'
        )
data_long2$case=factor(data_long2$case,levels=c('Experiment 1','Experiment 2','Experiment 3','Experiment 4'))
p2=ggplot(data_long2,aes(x=SNR,y=value,group=Method))+
  geom_line(aes(color=Method),size=1)+
  geom_point(aes(color=Method,shape=Method))+
  #scale_linetype_manual(values=c("twodash", "dotted"))+
  #theme_few(base_size = 14)+
  
  facet_wrap(~case, scales="free_y", nrow = 1)+#,labeller = as_labeller(label)) +
  scale_x_log10(labels = scientific_10)+
  theme(axis.title.x=element_text(size = 13),
        # axis.text.x = element_text(angle = 30),
        axis.text.x = element_text(),
        axis.title.y=element_blank(),
        #axis.text.y=element_blank(),
        #legend.position = "none"
        legend.position = "bottom",
        legend.title=element_text(size=13),
        legend.text=element_text(size=13),
        #plot.title = element_text(size = 40, face = "bold"),
        strip.text = element_text(size=13)
  )
p2
ggsave(filename = 'Time_of_4_experiment.eps',dpi=300,width=10.032,height = 7.464)
ggsave(filename = 'Time_of_4_experiment.png',dpi=300,width=15,height = 5)

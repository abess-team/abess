library(mccr)

n=1000
p=100  #20,30,40
k=40
testtms = 100

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


pe_gpdas_low=rep(0,testtms)
tpr_gpdas_low=rep(0,testtms)
fpr_gpdas_low=rep(0,testtms)
norm_gpdas_low=rep(0,testtms)
t_gpdas_low=rep(0,testtms)
mcc_gpdas_low=rep(0,testtms)
nz_gpdas_low=rep(0,testtms)


for(i in 1:testtms){
  # b=10*sqrt(2*log(p)/n)
  # B=5*b
  # beta_nonzero=runif(k,b,B)
  # beta=rep(0,p)
  # for(j in 1:k){
  #   beta[j]=beta_nonzero[j]
  # }

  data=gen.data(n,p,family = "gaussian" ,K=k,rho=0.5,sigma=3)
  
  beta = data$Tbeta
  subset=ifelse(beta==0,0,1)
  
  fit <- bess(data$x, data$y, s.list = 1:min(p, n/2), method = "sequential",
              family = "gaussian")
  fit2 <- bess(data$x, data$y, s.min=1, s.max=min(p, n/2), method = "gsection",
              family = "gaussian")
  
  # cvob_lasso=cv.glmnet(data$x,data$y, family='cox')
  # beta_lasso_temp=predict(cvob_lasso,type='coefficients',s=cvob_lasso$lambda.min)
  
  # cvob_ela=cv.glmnet(data$x,data$y,alpha=0.5,family='cox')
  # beta_ela_temp=predict(cvob_ela,type='coefficients',s=cvob_ela$lambda.min)
  ##############
  # pe_l0l2_low[i]=norm(y_test-(X_test%*%list_l0l2$beta+list_l0l2$coef0),type = '2')^2/norm(y_test,type='2')^2
  # # norm_l0l2_low[i]=norm(c(list_l0l2$coef0,list_l0l2$beta-beta),type='2')
  # mcc_l0l2_low[i]=mccr(subset,ifelse(list_l0l2$beta==0,0,1))
  # tpr_l0l2_low[i]=sum(ifelse(beta!=0&list_l0l2$beta!=0,1,0))/k
  # fpr_l0l2_low[i]=sum(ifelse(beta==0&list_l0l2$beta!=0,1,0))/(p-k)
  # nz_l0l2_low[i]=length(which(list_l0l2$beta!=0))/k
  
  # pe_pdas_low[i]=norm(y_test-(X_test%*%list_pdas$beta+list_pdas$coef0),type = '2')^2/norm(y_test,type='2')^2
  norm_pdas_low[i]=norm(fit$beta[,which.min(fit$EBIC)]-beta,type='2')
  mcc_pdas_low[i]=mccr(subset,ifelse(fit$beta[,which.min(fit$EBIC)]==0,0,1))
  tpr_pdas_low[i]=sum(ifelse(beta!=0&fit$beta[,which.min(fit$EBIC)]!=0,1,0))/k
  fpr_pdas_low[i]=sum(ifelse(beta==0&fit$beta[,which.min(fit$EBIC)]!=0,1,0))/(p-k)
  nz_pdas_low[i]=length(which(fit$beta[,which.min(fit$EBIC)]!=0))/k
  
  
  # pe_gpdas_low[i]=norm(y_test-(X_test%*%list_gpdas$beta+list_gpdas$coef0),type = '2')^2/norm(y_test,type='2')^2
  norm_gpdas_low[i]=norm(fit2$beta[,which.min(fit2$EBIC)]-beta,type='2')
  mcc_gpdas_low[i]=mccr(subset,ifelse(fit2$beta[,which.min(fit2$EBIC)]==0,0,1))
  tpr_gpdas_low[i]=sum(ifelse(beta!=0&fit2$beta[,which.min(fit2$EBIC)]!=0,1,0))/k
  fpr_gpdas_low[i]=sum(ifelse(beta==0&fit2$beta[,which.min(fit2$EBIC)]!=0,1,0))/(p-k)
  nz_gpdas_low[i]=length(which(fit2$beta[,which.min(fit2$EBIC)]!=0))/k
  
  # pe_ela_low[i]=norm(y_test-predict(cvob_ela,s=cvob_ela$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
  # norm_ela_low[i]=norm(beta_ela_temp-beta,type='2')
  # mcc_ela_low[i]=mccr(subset,ifelse(beta_ela_temp==0,0,1))
  # tpr_ela_low[i]=sum(ifelse(beta!=0&beta_ela_temp!=0,1,0))/k
  # fpr_ela_low[i]=sum(ifelse(beta==0&beta_ela_temp!=0,1,0))/(p-k)
  # nz_ela_low[i]=length(which(beta_ela_temp!=0))/k
  
  # pe_lasso_low[i]=norm(y_test-predict(cvob_lasso,s=cvob_lasso$lambda.min,newx = X_test),type = '2')^2/norm(y_test,type='2')^2
  # norm_lasso_low[i]=norm(beta_lasso_temp-beta,type='2')
  # mcc_lasso_low[i]=mccr(subset,ifelse(beta_lasso_temp==0,0,1))
  # tpr_lasso_low[i]=sum(ifelse(beta!=0&beta_lasso_temp!=0,1,0))/k
  # fpr_lasso_low[i]=sum(ifelse(beta==0&beta_lasso_temp!=0,1,0))/(p-k)
  # nz_lasso_low[i]=length(which(beta_lasso_temp!=0))/k
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


mean_pe_gpdas_low=mean(pe_gpdas_low)
mean_tpr_gpdas_low=mean(tpr_gpdas_low)
mean_fpr_gpdas_low=mean(fpr_gpdas_low)
mean_t_gpdas_low=mean(t_gpdas_low)
mean_norm_gpdas_low=mean(norm_gpdas_low)
mean_nz_gpdas_low=mean(nz_gpdas_low)
mean_mcc_gpdas_low=mean(mcc_gpdas_low)


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

data1=data.frame(mean_pe_lasso_low, mean_pe_l0l2_low,mean_pe_ela_low,mean_pe_pdas_low,mean_pe_gpdas_low,
                 mean_tpr_lasso_low,mean_tpr_l0l2_low,mean_tpr_ela_low,mean_tpr_pdas_low,mean_tpr_gpdas_low,
                 mean_fpr_lasso_low,mean_fpr_l0l2_low,mean_fpr_ela_low,mean_fpr_pdas_low,mean_fpr_gpdas_low,
                 mean_t_lasso_low,mean_t_l0l2_low,mean_t_ela_low,mean_t_pdas_low,mean_t_gpdas_low,
                 mean_norm_lasso_low,mean_norm_l0l2_low,mean_norm_ela_low,mean_norm_pdas_low,mean_norm_gpdas_low,
                 mean_nz_lasso_low,mean_nz_l0l2_low,mean_nz_ela_low,mean_nz_pdas_low,mean_nz_gpdas_low,
                 mean_mcc_lasso_low,mean_mcc_l0l2_low,mean_mcc_ela_low,mean_mcc_pdas_low,mean_mcc_gpdas_low)
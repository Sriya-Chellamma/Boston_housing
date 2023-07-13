setwd("C:/Users/Ramachandran/Desktop/Tableau Docs- BBL/Boston Housing")
hs=read.csv("train.csv")
library(ggplot2)
library(tidymodels)
library(themis)
library(workflowsets)

str(hs)
summary(hs)
hs$SalePrice=as.numeric(hs$SalePrice)
hist(hs$SalePrice)
ggplot(hs,aes(hs$MSZoning,hs$SalePrice))+geom_bar(stat = "identity")
table(hs$MSZoning)
ggplot(hs,aes(hs$LotFrontage,hs$SalePrice))+geom_point()
ggplot(hs,aes(hs$LotArea,hs$SalePrice))+geom_point()
ggplot(hs,aes(hs$Street,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$Alley,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$LotShape,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$LandContour,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$Utilities,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$LotConfig,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$LandSlope,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$Neighborhood))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$Condition1,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$Condition2,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$BldgType,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$HouseStyle,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$OverallQual,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$OverallCond,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$YearBuilt,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$YearRemodAdd,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$RoofStyle,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$RoofMatl,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$Exterior1st))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$Exterior2nd))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$MasVnrType))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$ExterQual))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$ExterCond))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$Foundation))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$BsmtQual))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$BsmtCond))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$BsmtExposure))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$BsmtFinType1))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$SalePrice,hs$BsmtFinSF1))+geom_point()
ggplot(hs,aes(hs$SalePrice,hs$BsmtFinType2))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$BsmtFinSF2,hs$SalePrice))+geom_point()
ggplot(hs,aes(hs$BsmtUnfSF,hs$SalePrice))+geom_point()
ggplot(hs,aes(hs$TotalBsmtSF,hs$SalePrice))+geom_point()
ggplot(hs,aes(hs$Heating,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$HeatingQC,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$CentralAir,hs$SalePrice))+geom_bar(stat = "identity")
ggplot(hs,aes(hs$Electrical,hs$SalePrice))+geom_bar(stat = "identity")
dim(hs)

hsFolds = vfold_cv(hs, v = 2,strata = hs$SalePrice)
myfolds2=vfold_cv(hs,v=2,strata = SalePrice)
hscaret= rsample2caret(myfolds2)

hsrecip = hs %>% recipe(SalePrice~.) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  step_nzv(all_predictors(),-all_outcomes()) %>% 
  step_corr(all_numeric(),threshold = 0.75)
hsrecip
hsxgbspec=boost_tree(mtry = tune(),sample_size = tune(),tree_depth = tune(),
                     trees = 500,learn_rate = tune(),loss_reduction = tune(),
                     min_n = tune()) %>% set_mode("regression") %>% 
  set_engine("xgboost")

hsxgb_params=parameters(list(min_n(),tree_depth(),learn_rate(),loss_reduction(),
                             sample_size=sample_prop(),finalize(mtry(),hs)))

hsxgbtree_grid=grid_latin_hypercube(hsxgb_params,size = 10)

hsmodels=workflow_set(preproc = list(hsrecip),
                      models = list(xgbTree=hsxgbspec),
                      cross = TRUE) %>% 
  option_add(grid=hsxgbtree_grid,id="recipe_xgbTree")
hsmodels

hsmetrics=metric_set(rmse,rsq)

hsmodelsrace= hsmodels %>% workflow_map("tune_grid",resamples=myfolds2,verbose = TRUE,
                                    control=control_grid(verbose = TRUE),
                                    metrics=hsmetrics)
hsmodelsrace %>% collect_metrics(metrics=hsmetrics)
autoplot(hsmodelsrace)
hsresults=hsmodelsrace %>% extract_workflow_set_result("recipe_xgbTree")
bestresults=hsresults %>% select_best(metric = "rmse")

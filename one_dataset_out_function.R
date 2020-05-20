
library(factoextra)
library(MASS)
library(dplyr)
library(data.table)
library(EloML)
library(tidyr)
library(ggplot2)
library(stringi)
library(grid)
library(gbm)
library(modelStudio)
library(purrr)
library(glmnet)
library(DALEXtra)
library(patchwork)
library(auditor)
library(ingredients)

one_set_out_cross <- function(all_train_data, target = 'avg_ranking', reg_formula, used_columns){
  
  dataset_unique <- unique(all_train_data$dataset)
  
  list_train_df <- map(dataset_unique,
                       ~all_train_data %>% filter(!dataset %in% .x))
  
  list_test_df <- map(dataset_unique,
                       ~all_train_data %>% filter(dataset %in% .x))
  
  MSE_TRAIN_EMPTY_MODEL <- map(list_train_df,
                             ~sum((.x[[target]] - mean(.x[[target]]))^2)/nrow(.x))
  MSE_TEST_EMPTY_MODEL <- map(list_test_df,
                               ~sum((.x[[target]] - mean(.x[[target]]))^2)/nrow(.x))
  
  ### train different type of surrogate models
  #browser()
  ##LASSO
  list_train_df_numeric <- map(list_train_df,
                               ~mutate_all(.x, as.numeric))
  list_test_df_numeric <- map(list_test_df,
                               ~mutate_all(.x, as.numeric))
  surr_CV_GLM_gr <- map(list_train_df_numeric,
                         ~cv.glmnet(x = as.matrix(.x[,used_columns]), y = as.matrix(.x[,target])))
  surr_GLM_gr <- map(list_train_df_numeric,
                     ~glmnet(x = as.matrix(.x[,used_columns]), y = as.matrix(.x[,target])))

  lambda_min_CV_GLM <- map(surr_CV_GLM_gr, ~.x$lambda.min)
  lambda_1se_CV_GLM <- map(surr_CV_GLM_gr, ~.x$lambda.1se)
  
  print('GLM fitted')
  
  ## GBM shallow
  surr_GBM_shallow_gr <- map(list_train_df,
                              ~gbm::gbm(reg_formula,
                                        distribution = 'gaussian',
                                        data = .x,
                                        n.trees = 30,
                                        interaction.depth = 2,
                                        n.minobsinnode = 4
                                        # ,train.fraction = 0.75
                                        , cv.folds = 4
                              ))
  print('GBM shallow fitted')
  ## GBM deep
  surr_GBM_deep_gr <- map(list_train_df,
                              ~gbm::gbm(reg_formula,
                                        distribution = 'gaussian',
                                        data = .x,
                                        n.trees = 30,
                                        interaction.depth = 10,
                                        n.minobsinnode = 4
                                        # ,train.fraction = 0.75 
                                        , cv.folds = 4
                              ))
  print('GBM deep fitted')
  ### prediction on test dataset
  
  ## LASSO
  test_pred_surr_CV_GLM_min_lambda <- pmap(.l =list(surr_CV_GLM_gr,
                                                    lambda_min_CV_GLM,
                                                    list_test_df_numeric),
                                            .f = ~predict(object = ..1, s =..2, newx = as.matrix(..3[,used_columns])))
  test_pred_surr_CV_GLM_1se_lambda <- pmap(.l =list(surr_CV_GLM_gr,
                                                    lambda_1se_CV_GLM,
                                                    list_test_df_numeric),
                                           .f = ~predict(object = ..1, s =..2, newx = as.matrix(..3[,used_columns])))
  train_pred_surr_CV_GLM_min_lambda <- pmap(.l =list(surr_CV_GLM_gr,
                                                    lambda_min_CV_GLM,
                                                    list_train_df_numeric),
                                           .f = ~predict(object = ..1, s =..2, newx = as.matrix(..3[,used_columns])))
  train_pred_surr_CV_GLM_1se_lambda <- pmap(.l =list(surr_CV_GLM_gr,
                                                    lambda_1se_CV_GLM,
                                                    list_train_df_numeric),
                                           .f = ~predict(object = ..1, s =..2, newx = as.matrix(..3[,used_columns])))
  
  
  ## GBM shallow
  test_pred_surr_GBM_shallow_gr <- map2(surr_GBM_shallow_gr,
                                        list_test_df,
                                        ~predict(.x, .y,
                                                 n.trees = gbm.perf(.x)))
  ## GBM deep
  test_pred_surr_GBM_deep_gr <- map2(surr_GBM_deep_gr,
                                        list_test_df,
                                        ~predict(.x, .y,
                                                 n.trees = gbm.perf(.x)))
  ### MSE on train data
  
  ## LASSO
  RATIO_MSE_TRAIN_surr_GLM_min_lambda <- map2(map2(list_train_df, train_pred_surr_CV_GLM_min_lambda, ~mean((.x[[target]] - .y)^2)),
                                              MSE_TRAIN_EMPTY_MODEL,
                                              ~.x/.y-1)
  RATIO_MSE_TRAIN_surr_GLM_1se_lambda <- map2(map2(list_train_df, train_pred_surr_CV_GLM_1se_lambda, ~mean((.x[[target]] - .y)^2)),
                                              MSE_TRAIN_EMPTY_MODEL,
                                              ~.x/.y-1)
  
  
  ## GBM shallow
  RATIO_MSE_TRAIN_surr_GBM_shallow_gr <- map2(surr_GBM_shallow_gr,
       MSE_TRAIN_EMPTY_MODEL,
       ~min(.x$train.error)/.y-1)
  
  ## GBM deep
  RATIO_MSE_TRAIN_surr_GBM_deep_gr <- map2(surr_GBM_deep_gr,
       MSE_TRAIN_EMPTY_MODEL,
       ~min(.x$train.error)/.y-1)
  
  ### MSE on test data
  
  ## LASSO
  RATIO_MSE_TEST_surr_GLM_min_lambda <- map2(map2(list_test_df, test_pred_surr_CV_GLM_min_lambda, ~mean((.x[[target]] - .y)^2)),
                                              MSE_TEST_EMPTY_MODEL,
                                              ~.x/.y-1)
  RATIO_MSE_TEST_surr_GLM_1se_lambda <- map2(map2(list_test_df, test_pred_surr_CV_GLM_1se_lambda, ~mean((.x[[target]] - .y)^2)),
                                              MSE_TEST_EMPTY_MODEL,
                                              ~.x/.y-1)
  MSE_TEST_surr_GLM_min_lambda <- map2(list_test_df, test_pred_surr_CV_GLM_min_lambda, ~mean((.x[[target]] - .y)^2))
  MSE_TEST_surr_GLM_1se_lambda <- map2(list_test_df, test_pred_surr_CV_GLM_1se_lambda, ~mean((.x[[target]] - .y)^2))
  
  ## GBM shallow
  RATIO_MSE_TEST_surr_GBM_shallow_gr <- map2(map2(list_test_df,test_pred_surr_GBM_shallow_gr, ~sum((.x[[target]] - .y)^2)/nrow(.x)),
                                              MSE_TEST_EMPTY_MODEL,
                                              ~.x/.y-1)
  MSE_TEST_surr_GBM_shallow_gr <- map2(list_test_df,test_pred_surr_GBM_shallow_gr, ~sum((.x[[target]] - .y)^2)/nrow(.x))
  
  ## GBM deep
  RATIO_MSE_TEST_surr_GBM_deep_gr <- map2(map2(list_test_df, test_pred_surr_GBM_deep_gr, ~sum((.x[[target]] - .y)^2)/nrow(.x)),
                                          MSE_TEST_EMPTY_MODEL,
                                          ~.x/.y-1)
  MSE_TEST_surr_GBM_deep_gr <- map2(list_test_df, test_pred_surr_GBM_deep_gr, ~sum((.x[[target]] - .y)^2)/nrow(.x))
  
  ### explainer
  # browser()
  ## LASSO
  
  explainer_GLM_lambda_min <- purrr::pmap(.l = list(surr_CV_GLM_gr,
                                                    list_train_df_numeric,
                                                    lambda_min_CV_GLM,
                                                    list_test_df),
                                          .f = ~DALEX::explain(..1,
                                                               data = as.matrix(..2[,used_columns]),
                                                               y = as.matrix(..2[,target]),
                                                               predict_function =  function(model, newdata){predict(model, newdata, s= 'lambda.min' )},
                                                               label = paste('Datset out:', unique(..4[,'dataset']), "Markers MODEL:LASSO_LAMBDA:Min_EXP:GBM"),
                                                               verbose = FALSE
                                          ))

  explainer_GLM_lambda_1se <- purrr::pmap(.l = list(surr_CV_GLM_gr,
                                                    list_train_df_numeric,
                                                    lambda_1se_CV_GLM,
                                                    list_test_df),
                                          .f = ~DALEX::explain(..1,
                                                               data = as.matrix(..2[,used_columns]),
                                                               y = as.matrix(..2[,target]),
                                                               predict = function(model, newdata){predict(model, newdata, s= 'lambda.1se' )[,1]},
                                                               label = paste('Datset out:', unique(..4[,'dataset']), "Markers MODEL:LASSO_LAMBDA:1se_EXP:GBM"),
                                                               verbose = FALSE
                                          ))
  
  
  ## GBM shallow
  explainer_surr_GBM_shallow_gr <- pmap(list(surr_GBM_shallow_gr,
                                             list_train_df,
                                             list_test_df),
                                              ~DALEX::explain(..1,
                                                              data = ..2[,used_columns],
                                                              y = ..2[,target],
                                                              predict = function(model, x){predict(model, x, n.trees = gbm.perf(model, plot.it = FALSE))},
                                                              label = paste('Datset out:', unique(..3[,'dataset']), "Markers MODEL:GBM_EXP:GBM_ntree:30_deepth_2"), verbose = FALSE))
  ## GBM deep
  explainer_surr_GBM_deep_gr <- pmap(list(surr_GBM_deep_gr,
                                          list_train_df,
                                          list_test_df),
                                        ~DALEX::explain(..1,
                                                        data = ..2[,used_columns],
                                                        y = ..2[,target],
                                                        predict = function(model, x){predict(model, x, n.trees = gbm.perf(model, plot.it = FALSE))},
                                                        label = paste('Datset out:', unique(..3[,'dataset']), "Markers MODEL:GBM_EXP:GBM_ntree:30_deepth_10"), verbose = FALSE))
  
  
  ### correlation spearman
  
  ## LASSO
  corr_spearman_test_GLM_lambda_min <- map2(list_test_df,
                                            test_pred_surr_CV_GLM_min_lambda,
                                         ~cor(.x[[target]],
                                              .y,
                                              method = 'spearman'))
  p_value_corr_spearman_test_GLM_lambda_min <- map2(list_test_df,
                                                    test_pred_surr_CV_GLM_min_lambda,
                                                 ~cor.test(.x[[target]],
                                                           .y,
                                                           method = 'spearman'))
  corr_spearman_test_GLM_lambda_1se <- map2(list_test_df,
                                            test_pred_surr_CV_GLM_1se_lambda,
                                            ~cor(.x[[target]],
                                                 .y,
                                                 method = 'spearman'))
  p_value_corr_spearman_test_GLM_lambda_1se <- map2(list_test_df,
                                                    test_pred_surr_CV_GLM_1se_lambda,
                                                    ~cor.test(.x[[target]],
                                                              .y,
                                                              method = 'spearman'))
  
  
  
  ## GBM shallow
  
  corr_spearman_test_GBM_shallow <- map2(list_test_df,
                                            test_pred_surr_GBM_shallow_gr,
                                            ~cor(.x[[target]],
                                                  .y,
                                                  method = 'spearman'))
  p_value_corr_spearman_test_GBM_shallow <- map2(list_test_df,
                                            test_pred_surr_GBM_shallow_gr,
                                            ~cor.test(.x[[target]],
                                                  .y,
                                                  method = 'spearman'))
  
  ## GBM deep
  corr_spearman_test_GBM_deep <- map2(list_test_df,
                                            test_pred_surr_GBM_deep_gr,
                                            ~cor(.x[[target]],
                                                  .y,
                                                  method = 'spearman'))
  p_value_corr_spearman_test_GBM_deep <- map2(list_test_df,
                                                    test_pred_surr_GBM_deep_gr,
                                                    ~cor.test(.x[[target]],
                                                               .y,
                                                               method = 'spearman'))
  
  return(list(test_df = list_test_df,
              prediction_GLM_lambda_min = test_pred_surr_CV_GLM_min_lambda,
              prediction_GLM_lambda_1se = test_pred_surr_CV_GLM_1se_lambda,
              prediction_GBM_shallow = test_pred_surr_GBM_shallow_gr,
              prediction_GBM_deep = test_pred_surr_GBM_deep_gr,
              
              explainer_GLM_lambda_min = explainer_GLM_lambda_min,
              explainer_GLM_lambda_1se = explainer_GLM_lambda_1se,
              explainer_GBM_shallow = explainer_surr_GBM_shallow_gr,
              explainer_GBM_deep = explainer_surr_GBM_deep_gr,
              
              mse_test_empty_model = MSE_TEST_EMPTY_MODEL,
              
              mse_test_glm_lambda_min = MSE_TEST_surr_GLM_min_lambda,
              mse_test_glm_lambda_1se = MSE_TEST_surr_GLM_1se_lambda,
              mse_test_gbm_shallow = MSE_TEST_surr_GBM_shallow_gr,
              mse_test_gbm_deep = MSE_TEST_surr_GBM_deep_gr,
              
              ratio_mse_train_glm_lambda_min = RATIO_MSE_TRAIN_surr_GLM_min_lambda,
              ratio_mse_train_glm_lambda_1se = RATIO_MSE_TRAIN_surr_GLM_1se_lambda,
              ratio_mse_train_gbm_shallow = RATIO_MSE_TRAIN_surr_GBM_shallow_gr,
              ratio_mse_train_gbm_deep = RATIO_MSE_TRAIN_surr_GBM_deep_gr,
              
              ratio_mse_test_glm_lambda_min = RATIO_MSE_TEST_surr_GLM_min_lambda,
              ratio_mse_test_glm_lambda_1se = RATIO_MSE_TEST_surr_GLM_1se_lambda,
              ratio_mse_test_gbm_shallow = RATIO_MSE_TEST_surr_GBM_shallow_gr,
              ratio_mse_test_gbm_deep = RATIO_MSE_TEST_surr_GBM_deep_gr,
              
              corr_spearman_test_GLM_lambda_min = corr_spearman_test_GLM_lambda_min,
              corr_spearman_test_GLM_lambda_1se = corr_spearman_test_GLM_lambda_1se,
              corr_spearman_test_GBM_shallow = corr_spearman_test_GBM_shallow,
              corr_spearman_test_GBM_deep = corr_spearman_test_GBM_deep,
              
              p_value_corr_spearman_test_GLM_lambda_min = p_value_corr_spearman_test_GLM_lambda_min,
              p_value_corr_spearman_test_GLM_lambda_1se = p_value_corr_spearman_test_GLM_lambda_1se,
              p_value_corr_spearman_test_GBM_shallow = p_value_corr_spearman_test_GBM_shallow,
              p_value_corr_spearman_test_GBM_deep = p_value_corr_spearman_test_GBM_deep,
              
              lambda_min_CV_GLM = lambda_min_CV_GLM,
              lambda_1se_CV_GLM = lambda_1se_CV_GLM 
              
              ))
}
  



















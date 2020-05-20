## meta-response
df_tidy_results_sample_100_gbm <- read.csv( './experiment_results/ranking_gbm_models_100.csv')[,-1]

# meta-features
param_gbm_def_rbind <- read.csv('./experiment_results/meta_features_hyperparameters.csv')[,-1]
properties_common_for_dataset <- read.csv('./experiment_results/meta_features_datasets.csv')[,-1]
landmarkers <- read.csv( './experiment_results/meta_features_landmarkers.csv')[,-1]

model_data <- df_tidy_results_sample_100_gbm %>% 
  left_join(param_gbm_def_rbind) %>% 
  left_join(properties_common_for_dataset) %>% 
  left_join(landmarkers) %>% 
  filter(group_dataset == 'continuous')

check_cardinality_columns <-apply(model_data, MARGIN =  2, FUN = function(x) length(unique(x)))
constant_columns <- names(check_cardinality_columns[check_cardinality_columns==1])


check_missings_columns <-apply(model_data, MARGIN =  2, FUN = function(x) mean(is.na(x)))
empty_columns <- names(check_missings_columns[check_missings_columns>0])

meta_features_list <- setdiff(colnames(model_data), unique(c(constant_columns, empty_columns, 'dataset', 'param_index', 'avg_perc_ranking_model', 'model_param')))


formula_meta_model <- as.formula(paste0('avg_perc_ranking_model~', paste0(meta_features_list, collapse = '+')))


## function to one-data-set-out crossvalidation
source('./one_dataset_out_function.R')

results_surrogate_model <- one_set_out_cross(all_train_data = model_data,
                                       target = 'avg_perc_ranking_model',
                                       reg_formula = formula_meta_model,
                                       used_columns = meta_features_list)

saveRDS(results_surrogate_model, 'summary_results_surrogate_models_rank_per_algo.Rd')


## in this object was save test data sets and results and explainers for surrogate models: GLM, GBM_shallow and GBM_deep
## each of objects is a list with 20 elements correspond to single test data sets (crossvalidation one-dataset-out)

#length(summary_results$test_df)

summary_results <- readRDS('summary_results_surrogate_models_rank_per_algo.Rd')

## mse results for GBM_deep
summary_results$ratio_mse_train_gbm_deep

### model gbm for 11 dataset out

test_data_set <- 11
test_data_set_id<- unique(summary_results$test_df[[test_data_set]]$dataset)



plot_check_prediction <- data.frame(prediction = summary_results$prediction_GBM_deep[[test_data_set]],
                                    actual_ranking = summary_results$test_df[[test_data_set]]$avg_perc_ranking_model) %>% 
  ggplot(aes(x=prediction))+
  geom_point(aes(y=actual_ranking), alpha = 0.4, col = 'red')+
  geom_line(aes(y=prediction))+
  labs(x = 'Prediction', y = 'Actual ranking',
       title = paste('Spearman correlation:',round(summary_results$corr_spearman_test_GBM_deep[[k]], 2),
                     '\n p.value: ',signif(summary_results$p_value_corr_spearman_test_GBM_deep[[k]]$p.value, 4)))+
  theme_light()


### variable importance

HYPERPARAMETERS  <-  c('n.trees', 'shrinkage', 'interaction.depth', 'n.minobsinnode', 'bag.fraction')
LANDMARKERS  <-  c("glmnet_def_to_gbm_def",  "kknn_def_to_gbm_def", "randomForest_def_to_gbm_def", "ranger_def_to_gbm_def")
DATASET_PROPERTIES <- setdiff(meta_features_list, c(HYPERPARAMETERS,LANDMARKERS))


group_variables <- list(HYPERPARAMETERS = HYPERPARAMETERS,
                        LANDMARKERS =LANDMARKERS,
                        DATASET_PROPERTIES = DATASET_PROPERTIES)

f_imp_GBM_deep <- ingredients::feature_importance(summary_results$explainer_GBM_deep[[test_data_set]])
f_imp_GBM_deep_group <- ingredients::feature_importance(summary_results$explainer_GBM_deep[[test_data_set]],
                                                        variable_groups = group_variables)


plot(f_imp_GBM_deep, max_vars = 15)

plot(f_imp_GBM_deep_group)


## ceteris paribus

## ceteris paribus - average hyperparameters for dataset

data_to_cdp <- model_data%>%
  mutate(split = ifelse(dataset %in% test_data_set_id, 'test', 'train')) %>% 
  group_by(dataset, split) %>% 
  summarise_all(mean) %>% 
  ungroup()

cdp_GBM_deep_dataset_avg_train <- ingredients::ceteris_paribus(summary_results$explainer_GBM_deep[[test_data_set]],
                                                               new_observation = data_to_cdp[-test_data_set,])
cdp_GBM_deep_dataset_avg_test <- ingredients::ceteris_paribus(summary_results$explainer_GBM_deep[[test_data_set]],
                                                              new_observation = data_to_cdp[test_data_set,])


SEL_FEATURE <- 'n.trees'
plot(cdp_GBM_deep_dataset_avg_train,
     variables = SEL_FEATURE, alpha =0.8, color = 'grey')+
  show_profiles(cdp_GBM_deep_dataset_avg_test, variables = SEL_FEATURE, color = 'red')+
  show_aggregated_profiles(cluster_profiles(cdp_GBM_deep_dataset_avg_train,
                                            center = TRUE,
                                            k=3,
                                            variables = SEL_FEATURE), 
                           color = "_label_", size = 2)+
  scale_x_continuous(trans='log2')+
  labs(x = paste0('log of ', SEL_FEATURE))+
  scale_color_discrete(name = 'Ceteris paribus profile for group', labels=c('A', 'B', 'C'))


SEL_FEATURE <- 'shrinkage'
plot(cdp_GBM_deep_dataset_avg_train,
     variables = SEL_FEATURE, alpha =0.8, color = 'grey')+
  show_profiles(cdp_GBM_deep_dataset_avg_test, variables = SEL_FEATURE, color = 'red')+
  show_aggregated_profiles(cluster_profiles(cdp_GBM_deep_dataset_avg_train,
                                            center = TRUE,
                                            k=3,
                                            variables = SEL_FEATURE), 
                           color = "_label_", size = 2)+
  scale_x_continuous(trans='log2')+
  labs(x = paste0('log of ', SEL_FEATURE))+
  scale_color_discrete(name = 'Ceteris paribus profile for group', labels=c('A', 'B', 'C'))

SEL_FEATURE <- 'interaction.depth'
plot(cdp_GBM_deep_dataset_avg_train,
     variables = SEL_FEATURE, alpha =0.8, color = 'grey')+
  show_profiles(cdp_GBM_deep_dataset_avg_test, variables = SEL_FEATURE, color = 'red')+
  show_aggregated_profiles(cluster_profiles(cdp_GBM_deep_dataset_avg_train,
                                            center = TRUE,
                                            k=3,
                                            variables = SEL_FEATURE), 
                           color = "_label_", size = 2)+
  scale_x_continuous(trans='log2')+
  labs(x = paste0('log of ', SEL_FEATURE))+
  scale_color_discrete(name = 'Ceteris paribus profile for group', labels=c('A', 'B', 'C'))

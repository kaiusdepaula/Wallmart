library(data.table)
library(lightgbm)
library(ggplot2)

#Lista de parâmetros de treinamento do modelo
param <- list(boosting_type = "gbdt",
              objective = "tweedie",
              tweedie_variance_power = 1.1,
              metric = "rmse",
              subsample = 0.5,
              subsample_freq = 1,
              learning_rate = 0.015,
              num_leaves = 2**8-1,
              min_data_in_leaf = 2**8-1,
              feature_fraction = 0.5,
              max_bin = 100,
              n_estimators = 200,
              boost_from_average = FALSE,
              verbose = -1,
              seed = 1995)

lojas <- c("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", 
           "WI_2", "WI_3")

setwd("~/projetos/Wallmart/modelos")

#Essa parte nao é tão prática, primeio leia qualquer arquivo processado e use a função
source("~/projetos/Wallmart/Scripts/funcoes/get_cvs.R")
dataset <- as.data.table(arrow::read_parquet("~/projetos/Wallmart/dados_de_treinamento/data_processada_CA_1.parquet"))

#Retirando os dias de previsão para evaluation (private score)
dataset <- dataset[dia < 1942]

cvs <- get_cvs(dataset, splits = 1)
rm(dataset)

scores <- data.table()
for (loja in lojas) {
  scores_cv <- c()
  for (cv in cvs) {
    print(paste0("Treinando o modelo da loja ", loja, " durante ", cv[1], " (treino) ", cv[2], " (validação)"))
    #Leitura dos dados
    dataset <- as.data.table(arrow::read_parquet(paste0("~/projetos/Wallmart/dados_de_treinamento/data_processada_", loja, ".parquet")))
    
    #Retirando o periodo usado para prever
    dataset <- dataset[dia <= cv[[2]]]
    
    #Declarando tipo
    col_categoricas <- c(
      "store_id", "dept_id", "cat_id",
      "item_id", "event_name_1", "event_name_2", "event_type_2",
      "event_type_1", "snap_CA", "month", "weekday", 
      "yearday", "year"
    )
    dataset[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]
    
    #Colunas que vão sair do escopo de treinamento 
    xFEAT <- setdiff(names(dataset), c(
      "date", "sales", "ordenador", "state_id"
    ))
    
    #Treinamento
    dt_train <- dataset[dia <= cv[[1]]]
    
    dt_trainMT = as.matrix(dt_train[,lapply(.SD, as.numeric),.SDcols = xFEAT])
    dt_trainDM = lgb.Dataset(dt_trainMT, label = log1p(dt_train[["sales"]]))
    
    #Validação
    dt_valid <- dataset[dia > cv[[1]] & dia <= cv[[2]]]
    
    dt_validMT = as.matrix(dt_valid[,lapply(.SD, as.numeric),.SDcols = xFEAT])
    dt_validDM = list(test = lgb.Dataset.create.valid(dt_trainDM, dt_validMT, label = log1p(dt_valid[["sales"]])))
    
    # fit model
    model <- lgb.train(params = param, data = dt_trainDM, valid=dt_validDM)
    
    scores_cv <- rbind(scores_cv, model$best_score)
  }
  
  lgb.save(model, paste0("Modelo_teste_cv", loja, ".txt"))
  scores[, paste0(loja) := scores_cv]
}

#Salvando scores de cv
fwrite(scores, "~/projetos/Wallmart/Scores_CV/Primeira_vez.csv")

#Pegando scores de cv (rapido)
# scores <- fread("~/projetos/Wallmart/Scores_CV/Primeira_vez.csv")

for (col in names(scores)) {
  print(paste0("Score médio do modelo ", col))
  print(scores[, mean(get(col))])
}




#rascunho
data <- data.table(model$record_evals$test$rmse$eval)
data[, id:= 1:.N]
ggplot(data = data, aes(y = as.numeric(V1), x = id)) + geom_line()

importance <- lgb.importance(model)
lgb.plot.importance(importance, top_n = 20)

#Plot agregado
dt_valid[, predicao := expm1(predict(model, dt_validMT))]

dt_plot <- dt_valid[
  , 
  .(
    predicao = sum(predicao, na.rm = TRUE),
    actual = sum(sales, na.rm = TRUE),
    dept_id
  ), by = c("date")]


colors <- c("actual" = "darkred", "prediction" = "steelblue")
ggplot(data = dt_plot, aes (x = date)) + 
  geom_point(aes (y = actual, color = "actual")) +
  geom_point(aes (y = predicao, color = "prediction")) +
  labs(y = "Actual X Prediction") + theme_classic() +
  scale_color_manual(values = colors)  +
  facet_wrap(~ dept_id)

#diff Plot
dt_plot <- dt_valid[, difference := round(predicao - sales, 2), by = "item_id"]
ggplot(data = dt_valid, aes (x = date)) +
  geom_point(aes (y = difference, group = item_id)) +
  labs(y = "Actual - Prediction") + theme_classic() +
  scale_y_continuous(limit = c(-10, 10), breaks = c(-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10)) +
  facet_wrap(~ dept_id)

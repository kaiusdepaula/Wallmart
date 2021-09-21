rmsse <- function(preds, whole_data, fold = c("tradicional", "nao_tradicional")) {
  if (fold == "tradicional") {
    #Seleciona dados de treinamento
    data <- whole_data[tipo == 0, .(N, N1)] 
    
    RMSE_future <- ModelMetrics::rmse(whole_data[tipo == 2L, N], expm1(preds))
    
  } else if(fold == "nao_tradicional") {
    #Seleciona dados de treinamento
    data <- whole_data[fold != 1L & tipo == 0L, .(N, N1)]
    
    RMSE_future <- ModelMetrics::rmse(whole_data[fold == 1L & tipo == 2L, N], expm1(preds))
    
  }
  

  
  #Aqui, farei de forma bem simplificada o calculo do rmse historico
  
  #Calcula o multiplicador da soma das diferenças por periodo
  multiplicador <- (1 / (nrow(data) - 1))
  
  #Menos a primeira entrada
  data <- data[-1, ]
  
  #Calculo da diferença do periodo menos o periodo defasado em 1, ao quadrado
  data[, difference_squared := (N - N1) ** 2]
  
  #Soma valores multiplicados pelo multiplicador 
  RMSE_historical <- multiplicador * data[, sum(difference_squared, na.rm = TRUE)] #Denominador
  
  
  #Aqui, basta fazer a razão dos rmse, e calcular a raiz quadrada
  rmsse <- sqrt(RMSE_future / RMSE_historical)
 
  
  return(rmsse)
}
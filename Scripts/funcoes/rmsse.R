rmsse <- function(id) {
  
  data <- dt_train[, .(sales)] 
  data2 <- dt_valid
  preds <- predict(model, dt_validMT)
  
  RMSE_future <- ModelMetrics::rmse(dt_valid[, sales], expm1(preds)) #Numerador
  RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
  rmsse <- RMSE_future / RMSE_historical
 
  return(rmsse)
}
#Função de validação cruzada em serie de tempo
get_cvs <- function(dataset, splits = 4, pred_horizon = 28) {
  dias <- as.numeric(dataset[, max(dia)]) - pred_horizon
  proportion <- 1/splits
  
  #Gerando valores
  cvs <- data.table()
  contador <- 1
  while (contador != (splits + 1)) {
    tr = c(
      round(dias * (contador * proportion)))
    tt = c(
      round(dias * (contador * proportion) + pred_horizon))

    cvs[, paste0(contador) := c(tr, tt)]

    contador = contador + 1
  }
  return (cvs)
}
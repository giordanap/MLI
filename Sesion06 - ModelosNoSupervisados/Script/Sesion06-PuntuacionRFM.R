################################################################################
###########---------- Machine Learning Inmersion ------------------#############
################################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com / andre.chavez@urp.edu.pe
# Tema:  Puntuacion o Segmentacion RFM
# version: 1.0
#########################################################################

# ######### 1) Librerias a utilizar ################# 
library(rgl)

# Leemos el dataset
data <- read.table("ClientesRETAIL.txt", header=T)
str(data)
head(data)
# Ponemos las fechas en un formato que nos ayude a trabajarlas
data$Fechas <- as.Date.factor(data$Fechas, format = "%Y-%m-%d")
str(data)

# Creando la Recencia
data$Recencia <- round(as.numeric(difftime(Sys.Date(), data$Fechas, units="days")) )

# Creando las variables de 'Monto total', 'Frecuencia' y 'Recencia' 
# Creamos el monto total
dataM <- aggregate(data$Monto, list(data$Cliente), sum)
names(dataM) <- c("Cliente", "Monto")

# Creamos la frecuencia
dataF <- aggregate(data$Monto, list(data$Cliente), length)
names(dataF) <- c("Cliente", "Frecuencia")

# Creamos la recencia
dataR <- aggregate(data$Recencia, list(data$Cliente), min)
names(dataR) <- c("Cliente", "Recencia")

# Combinando las variables o puntuaciones de R, F y M
temp <- merge(dataF, dataR, "Cliente")
dataRFM <- merge(temp,dataM,"Cliente")

# Pasamos a crear los cortes de informacion (Esto es dependiendo de lo que estamos estudiando)
# Crea los niveles R,F,M (quintiles)
dataRFM$rankR <- cut(-(dataRFM$Recencia), 5, labels = F)  # 5 es el mas reciente, 1 el menos
dataRFM$rankF <- cut(dataRFM$Frecuencia, 5, labels = F)  
dataRFM$rankM <- cut(dataRFM$Monto, 5, labels = F)   

# Hacemos los cruces de informacion dependiendo de lo necesitemos mostrar.
table(dataRFM[,5:6])
table(dataRFM[,6:7])
table(dataRFM[,5:7])

# Podemos mostrar el TOP 10 de los clientes
dataRFM <- dataRFM[with(dataRFM, order(-rankR, -rankF, -rankM)), ]
head(dataRFM, n=10)

# Agrupando o creando la puntuacion de valor RFM
groupRFM <- dataRFM$rankR*100 + dataRFM$rankF*10 + dataRFM$rankM
dataRFM <- cbind(dataRFM,groupRFM)
head(dataRFM, n=10)

# Podemos mostrar graficas de la distribucion de la cartera por RFM
library(ggplot2)
ggplot(dataRFM, aes(factor(groupRFM))) + geom_bar(fill = "white", colour = "blue") + ggtitle('Segmentacion de Clientes RFM') + labs(x="RFM",y="#Clientes")

# Podemos mostrar graficos en 3D de donde se concentran nuestros mejores clientes
library(rgl)
plot3d( dataRFM$rankR, dataRFM$rankF, dataRFM$rankM, xlab = "Recencia",  ylab = "Frecuencia", zlab = "Monto",
        col = dataRFM$groupRFM, type = "s", radius = 0.5 )

# FIN !!
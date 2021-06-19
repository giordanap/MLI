#########################################################################
#########------- Machine Learning Inmersion ------------#################
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com
# Tema: Entendimiento y Preprocesamiento datos.
# version: 1.0
#########################################################################

#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


###########################################
######  CASO DREAM HOUSING FINANCE  #######
###########################################

#############################
#  1. Descripcion del caso  #
#############################

# Introduccion:
# La compañia Dream Housing Finance se ocupa de todos los 
# prestamos hipotecarios. Tiene presencia en todas las areas 
# urbanas, semi urbanas y rurales. 
# El cliente primero solicita un prestamo hipotecario y luego  
# la compañia valida si el cliente es un prospecto o no de darle
# un prestamo hipotecario.
#
# Problema:
# La empresa desea automatizar el proceso de elegibilidad del 
# prestamo (en tiempo real) en funcion del detalle del cliente
# al completar el formulario de solicitud. 
# Las variables que tiene la empresa para tal desafio son:

# Genero, Estado civil, Educacion, Numero de dependientes, 
# Ingresos, Monto del prestamo, historial de credito y otros. 
# 
# Para automatizar este proceso, se desea identificar los 
# grupos de clientes, que son elegibles para el monto del 
# prestamo para que puedan dirigirse especeficamente a estos 
# clientes. 

# Cargando los paquetes
library(caret)
library(DataExplorer)
library(VIM)
library(missForest)
library(ggplot2)
library(dummies)
library(iplots)

# Cargando los datos 
datos<-read.csv("loan_prediction-II.csv",
                stringsAsFactors = T, 
                sep=";",
                na.strings = "")

# Viendo la estructura de los datos
str(datos)

# Eliminando la columna de identificacion del cliente (Loan_ID)
datos$Loan_ID <- NULL

# Declarar la variable Credit_History como factor
datos$Credit_History <- as.factor(datos$Credit_History)

levels(datos$Credit_History)  <- c("Malo","Bueno")
str(datos)

#####################################
# 2. Pre-procesamiento de los datos #
#####################################

#----------------------------------------------------------
# Verificacion de datos perdidos

library(DataExplorer)
plot_missing(datos)

# Para ver las variables con valores perdidos 
which(colSums(is.na(datos))!=0)

# Para ver que filas tienen valores perdidos
rmiss=which(rowSums(is.na(datos))!=0,arr.ind=T)
rmiss

# Para ver cuantas filas tienen valores perdidos
length(rmiss)

# Para ver el porcentaje de filas con valores perdidos
length(rmiss)*100/dim(datos)[1]

# Total de datos perdidos
sum(is.na(datos))

# Graficar la cantidad de valores perdidos
library(VIM)
windows()
graf_perdidos1 <- aggr(datos,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

summary(graf_perdidos1)

matrixplot(datos,
           main="Matrix Plot con Valores Perdidos",
           cex.axis = 0.6,
           ylab = "registro")

hist(datos$ApplicantIncome) # Histograma de los datos
hist(datos$CoapplicantIncome)
hist(datos$LoanAmount)
hist(datos$Loan_Amount_Term)


#----------------------------------------------------------------
# Opcion 1
# Imputando los valores perdidos cuantitativos usando k-nn
# y estandarizando las variables numericas
library(caret)
library(RANN)
set.seed(123)
preProcValues1 <- preProcess(datos,
                             method=c("knnImpute","center","scale"))

# Otras opciones: range , bagImpute, medianImpute

preProcValues1

datos_transformado1 <- predict(preProcValues1, datos)

# Distribucion de las variables numericas transformadas
hist(datos_transformado1$ApplicantIncome)
hist(datos_transformado1$CoapplicantIncome)
hist(datos_transformado1$LoanAmount)
hist(datos_transformado1$Loan_Amount_Term)

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado1))

# Graficar la cantidad de valores perdidos en las 
# variables categoricas
graf_perdidos2 <- aggr(datos_transformado1,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

summary(graf_perdidos2)

# Imputacion de datos categoricos
table(datos_transformado1$Gender)
table(datos_transformado1$Gender,useNA="always")
table(datos_transformado1$Married,useNA="always")
table(datos_transformado1$Self_Employed,useNA="always")
table(datos_transformado1$Credit_History,useNA="always")

# Imputar valores missing usando el algoritmo Random Forest
library(missForest)
set.seed(123)
impu_cate          <- missForest(datos_transformado1)

datos_transformado1 <- impu_cate$ximp

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado1))

plot_missing(datos_transformado1)

# Identificando variables con variancia cero o casi cero
nearZeroVar(datos_transformado1, saveMetrics= TRUE)

#                    freqRatio percentUnique zeroVar   nzv
# Gender              7.832565    0.01739130   FALSE FALSE
# Married             3.561682    0.01739130   FALSE FALSE
# Dependents          2.941230    0.03478261   FALSE FALSE
# Education           3.195549    0.01739130   FALSE FALSE
# Self_Employed       8.511993    0.01739130   FALSE FALSE
# ApplicantIncome     1.024691    4.39130435   FALSE FALSE
# CoapplicantIncome   3.042683    3.40000000   FALSE FALSE
# LoanAmount          1.088161    1.84347826   FALSE FALSE
# Loan_Amount_Term   11.683589    0.09565217   FALSE FALSE
# Credit_History      5.788666    0.01739130   FALSE FALSE
# Property_Area       1.182395    0.02608696   FALSE FALSE
# Nacionality       337.235294    0.01739130   FALSE  TRUE
# Loan_Status         2.509307    0.01739130   FALSE FALSE


unique(datos_transformado1$LoanAmount)

table(datos_transformado1$Nacionality)
datos_transformado1$Nacionality <- NULL


# Verificando freqRatio y percentUnique para Gender
table(datos_transformado1$Gender)

# Female   Male 
#   1302  10198 

# freqRatio     = (10198/1302)   = 7.832565
# percentUnique = (2/11500)*100  = 0.01739130  

# Verificando la estructura del archivo pre-procesado
str(datos_transformado1)

#----------------------------------------------------------------
# Creando variables dummies

# Usando el paquete dummies
library(dummies)
datos_transformado1 <- dummy.data.frame(datos_transformado1,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))
# Verificando la estructura del archivo pre-procesado
str(datos_transformado1)

#----------------------------------------------------------------
# Opcion 2
# Imputando los valores perdidos cuantitativos usando k-nn
# y aplicando transformacion Box-Cox a las variables numericas
library(caret)
set.seed(123)
preProcValues2 <- preProcess(datos, method=c("knnImpute","BoxCox"))

preProcValues2

datos_transformado2 <- predict(preProcValues2, datos)

# Distribucion de las variables numericas transformadas
hist(datos_transformado2$ApplicantIncome)
hist(datos_transformado2$CoapplicantIncome)
hist(datos_transformado2$LoanAmount)
hist(datos_transformado2$Loan_Amount_Term)

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado2))

# Graficar la cantidad de valores perdidos en las 
# variables categoricas
graf_perdidos2 <- aggr(datos_transformado2,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

summary(graf_perdidos2)

# Imputacion de datos categoricos
# Imputar valores missing usando el algoritmo Random Forest
library(missForest)
set.seed(123)
impu_cate          <- missForest(datos_transformado2)
datos_transformado2 <- impu_cate$ximp

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado2))

plot_missing(datos_transformado2)

# Identificando variables con variancia cero o casi cero
nearZeroVar(datos_transformado2, saveMetrics= TRUE)


table(datos_transformado2$Nacionality)
datos_transformado2$Nacionality <- NULL


# Verificando la estructura del archivo pre-procesado
str(datos_transformado2)


#----------------------------------------------------------------
# Creando variables dummies

# Usando el paquete dummies
library(dummies)
datos_transformado2 <- dummy.data.frame(datos_transformado2,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))


#---------------------------------------------------
#  Identificando predictores correlacionados o colineales

descrCor <- cor(datos_transformado1[,c(13:16)])
descrCor

summary(descrCor[upper.tri(descrCor)])

altaCorr <- findCorrelation(descrCor, cutoff = .50, names=TRUE)
altaCorr

descrCor2 <- cor(datos_transformado1[,c(13,14,16)])
summary(descrCor2[upper.tri(descrCor2)])

datos_transformado1 <- datos_transformado1[,-15]
altaCorr2 <- findCorrelation(descrCor2, cutoff = .50, names=TRUE)
altaCorr2


###########################################
###########  CASO CENSUS  #################
###########################################

# Conjunto de 32561 observaciones provenientes de un censo 
# poblacional.
# El objetivo es poder predecir el salario de una persona de 
# manera categorica : <=50K o >50K

# Cargando los datos con la funci?n read.csv
census.csv <- read.csv("censusn.csv",sep=";")
str(census.csv)

# Cargando los datos con la funcion fread() de data.table
library(data.table)
censusn <-fread("censusn.csv",
                header=T, 
                verbose =FALSE, 
                stringsAsFactors=TRUE,
                showProgress =TRUE)

str(censusn)

save(censusn, file = "census_dt.Rdata") 
rm(censusn)
load("census_dt.Rdata")

#--------------------------------------------------------
# 1. Deteccion de valores perdidos

# Detecci?n de valores perdidos con el paquete DataExplorer
library(DataExplorer)
plot_missing(censusn) 

# Para ver las variables con valores perdidos 
which(colSums(is.na(censusn))!=0)

# Para ver cuantas filas tienen valores perdidos
rmiss <- which(rowSums(is.na(censusn))!=0,arr.ind=T)
length(rmiss)

# Para ver el porcentaje de filas con valores perdidos
length(rmiss)*100/dim(censusn)[1]

# Para graficar la cantidad de valores perdidos
library(VIM)
valores.perdidos <- aggr(censusn,numbers=T)
valores.perdidos
summary(valores.perdidos)

matrixplot(censusn,
           main="Matrix Plot con Valores Perdidos",
           cex.axis = 0.6,
           ylab = "registro")

#----------------------------------------------------
# 2. Eliminacion de datos perdidos
census.cl <- na.omit(censusn)
plot_missing(census.cl) 


#---------------------------------------------
# 3. Imputacion con el paquete DMwR
library(DMwR)

# Funcion centralImputation()
# Si la variable es numerica (numeric o integer) reemplaza los valores 
# faltantes con la mediana.
# Si la variable es categorica (factor) reemplaza los valores faltantes con 
# la moda. 

census.ci <-centralImputation(censusn)
plot_missing(census.ci) 

#----------------------------------------------------------------------
# 4. Imputar los datos usando mice para las cualitativas

library(mice)
set.seed(123)
mice_imputes = mice(censusn, m=2, maxit = 1, method = "cart")

# Datos imputados
censusn_transformado2 <- complete(mice_imputes,1)

# Verificando la cantidad de valores perdidos
sum(is.na(censusn_transformado2))

plot_missing(censusn_transformado2)


###########################
#  DETECCION DE OUTLIERS  #
###########################

#-------------------------------------- 
# 1. Deteccion de outliers univariados

boxplot(censusn$age,col="peru")

outliers1 <- boxplot(censusn$age)$out
outliers1 ; length(outliers1)
summary(outliers1)


boxplot(censusn$hours.per.week,col="peru")

outliers2 <- boxplot(censusn$hours.per.week)$out
outliers2 ; length(outliers2)
summary(outliers2)


##############################
#  TRANSFORMANDO VARIABLES   #
##############################

#--------------------------------------------------------------------
# 1. Transformando una variable numerica a categorica usando BoxPlot
# Ejemplo: hours.per.week

library(ggplot2)
summary(censusn$hours.per.week)
ggplot(aes(x = factor(0), y = hours.per.week),
       data = censusn) + 
  geom_boxplot() +
  stat_summary(fun.y = mean, 
               geom = 'point', 
               shape = 19,
               color = "red",
               cex = 2) +
  scale_x_discrete(breaks = NULL) +
  scale_y_continuous(breaks = seq(0, 100, 5)) + 
  xlab(label = "") +
  ylab(label = "Working Hours per week") +
  ggtitle("Box Plot of Working Hours per Week") 

summary(censusn$hours.per.week)

# Menos de 40
# De 40 a 45
# De 45 a m?s

censusn$hpw_cat1 <- cut(censusn$hours.per.week, 
                       breaks = c(-Inf,40,45,Inf),
                       labels = c("Menos de 40", "De 40 a menos de 45",
                                  "De 45 a mas"),
                     right = FALSE)

table(censusn$hpw_cat1)  

ggplot(censusn, aes(hpw_cat1)) + 
  geom_bar(color="blue",fill="darkgreen") + 
  theme_light() + 
  labs(title ="Gr?fico de Barras", 
       x="Horas de trabajo a la semana", 
       y= "Frecuencia") 


#---------------------------------------------------------------------
# 2. Transformando una variable numerica a categorica usando iplots
# Ejemplo: hours.per.week

library(iplots)
ibar(censusn$salary)
ihist(censusn$hours.per.week)

# Menos de 40
# De 40 a mas

censusn$hpw_cat2 <- cut(censusn$hours.per.week, 
                       breaks = c(-Inf,40,Inf),
                       labels = c("Menos de 40","De 40 a mas"),
                       right = FALSE)

table(censusn$hpw_cat2)  

ggplot(censusn, aes(hpw_cat2)) + 
  geom_bar(color="blue",fill="orange") + 
  theme_light() + 
  labs(title ="Gr?fico de Barras", 
       x="Horas de trabajo a la semana", 
       y= "Frecuencia")


#--------------------------------------------------------------------
# 3. Transformando una variable numerica a categorica usando arboles
# Ejemplo: hours.per.week

library(rpart)

set.seed(123)
arbol <- rpart(salary ~ hours.per.week ,
                        data=censusn,
                        method="class",
                        control=rpart.control(cp=0,minbucket=0)
                        )

library(rpart.plot)
rpart.plot(arbol, 
           digits=-1,
           type=2, 
           extra=101,
           varlen = 3,
           cex = 0.7, 
           nn=TRUE)


# Menos de 42
# De 42 a mas

censusn$hpw_cat3 <- cut(censusn$hours.per.week, 
                        breaks = c(-Inf,42,Inf),
                        labels = c("Menos de 42","De 42 a mas"),
                        right = FALSE)

table(censusn$hpw_cat3)  

ggplot(censusn, aes(hpw_cat3)) + 
  geom_bar(color="blue",fill="orange") + 
  theme_light() + 
  labs(title ="Grafico de Barras", 
       x="Horas de trabajo a la semana", 
       y= "Frecuencia") 

# FIN!
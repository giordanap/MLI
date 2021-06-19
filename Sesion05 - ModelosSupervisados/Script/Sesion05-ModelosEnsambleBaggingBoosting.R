#########################################################################
#########------- Machine Learning Inmersion ------------#################
#########################################################################

# Capacitador: Andr√© Omar Ch√°vez Panduro
# email: andrecp38@gmail.com
# Modelos de Ensamble: Algoritmos de Bagging - Boosting
# version: 1.0
#########################################################################

#################################################################
##### (1) DESARROLLO DE ALGORITMOS DE MACHINE LEARNING ##########
#################################################################
################################
#                              #
#  CASO LOAN PREDICTION III    #
#                              #
################################

#############################
#   Descripcion del caso  #
#############################

# Introduccion:
# La compa√±ia Dream Housing Finance se ocupa de todos los 
# prestamos hipotecarios. Tiene presencia en todas las areas 
# urbanas, semi urbanas y rurales. 
# El cliente primero solicita un prestamo hipotecario y luego  
# la compa√±ia valida si el cliente es un prospecto o no de darle
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

#####################################
####### 0. Librerias a utilizar #####
##################################### 

# Pacman nos ayuda a instalar multiples librerias
install.packages("pacman")
library(pacman)
p_load(dplyr, psych, tm,party,pillar,ggplot2,sqldf,ggvis,Boruta,pROC,
       randomForest,e1071,caret,glmnet,xgboost,ROCR,C50,mlr,lattice,
       gmodels,gplots,DMwR,rminer,polycor,class)

# Activar las librerias para su uso.
library(sqldf);library(ggvis);library(Boruta);
library(pROC);library(randomForest);library(caret);
library(glmnet);library(ROCR);library(mlr);# Mlr en R
library(gmodels);library(gplots);library(DMwR);
library(polycor);library(class);

#####################################
##### 1. Lectura de los datos #######
#####################################

setwd("D:/DMC/Desktop/Sesion05 - ModelosSupervisados/DataSet")
datosD<- read.csv("train.csv",na.strings = c(""," ",NA)) # Leer la data de entrenamiento

# Visualizar los nombres de la data
names(datosD) 

# Viendo la estructura de los datos
str(datosD)

# Eliminando la columna de identificacion del cliente (Loan_ID)
datosD$Loan_ID<-NULL

# Declarar la variable Credit_History como factor
datosD$Credit_History <- as.factor(datosD$Credit_History)

levels(datosD$Credit_History)  <- c("Malo","Bueno")
str(datosD)

#####################################
# 2. Pre-procesamiento de los datos #
#####################################

# Tablas resumen - Analisis univariado de la informacion

library(mlr)
summarizeColumns(datosD) # tabla mas completa
Resumen_Datos<-summarizeColumns(datosD) # Guardo en un objeto
write.csv(Resumen_Datos,"Resumen_1sesion.csv") # Escribir o sacar un objeto de R

#  Comentarios de la data

# 1. LoanAmount tiene (614 - 592) 22 valores perdidos.
# 2. Loan_Amount_Term tiene (614 - 600) 14 valores perdidos.
# 3. Credit_History tiene (614 - 564) 50 valores perdidos.
# 4. Nosotros podemos tambiEn observar que cerca del 84% de los solicitantes al prestamo 
# tienen un historial crediticio. Como La media del campo Credit_History es 0.84 
# (Recordemos, Credit_History tiene o toma el valor 1 para aquellos que tienen 
#   historial crediticio y 0 en caso contrario).
# 5. La variable ApplicantIncome parece estar en l?nea con las espectativas al 
# igual que CoapplicantIncome.

#----------------------------------------------------------
# Verificacion de datos perdidos
library(DataExplorer)
plot_missing(datosD)

# Graficar la cantidad de valores perdidos
library(VIM)
graf_perdidos1 <- aggr(datosD,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

summary(graf_perdidos1)

# Matriz de datos x variables perdidas
matrixplot(datosD,
           main="Matrix Plot con Valores Perdidos",
           cex.axis = 0.6,
           ylab = "registro")
#----------------------------------------------------------
#  Recodificacion de Variables para el Train
#  Recodificando Dependents
datosD$Dependents=ifelse(datosD$Dependents=="3+",3,
                         ifelse(datosD$Dependents=="0",0,
                                ifelse(datosD$Dependents=="1",1,
                                       ifelse(datosD$Dependents=="2",2,
                                              datosD$Dependents))))
datosD$Dependents=as.factor(datosD$Dependents)

#  Recodificando Gender
datosD$Gender=ifelse(datosD$Gender=="Male",1,0)
datosD$Gender=as.factor(datosD$Gender)


#  Recodificando Married
datosD$Married=ifelse(datosD$Married=="Yes",1,0)
datosD$Married=as.factor(datosD$Married)

#  Recodificando Education
datosD$Education=ifelse(datosD$Education=="Graduate",1,0)
datosD$Education=as.factor(datosD$Education)

#  Recodificando Self Employed
datosD$Self_Employed=ifelse(datosD$Self_Employed=="Yes",1,0)
datosD$Self_Employed=as.factor(datosD$Self_Employed)

#  Recodificando Property Area
datosD$Property_Area=ifelse(datosD$Property_Area=="Rural",0,
                            ifelse(datosD$Property_Area=="Semiurban",1,
                                   2))
datosD$Property_Area=as.factor(datosD$Property_Area)

#  Recodificando Credit History
datosD$Credit_History=ifelse(datosD$Credit_History=="Bueno",1,0)
datosD$Credit_History=as.factor(datosD$Credit_History)

#  Convirtiendo en factor el Target
datosD$Loan_Status=ifelse(datosD$Loan_Status=="Y",1,0)
datosD$Loan_Status <- as.factor(datosD$Loan_Status)

table(datosD$Credit_History)

#----------------------------------------------------------------
# Opcion 1 - Sofisticada o por Machine Learning
# Imputando los valores perdidos cuantitativos usando k-nn
# y estandarizando las variables numericas
library(caret)
set.seed(123)
preProcValues1 <- caret::preProcess(datosD,
                    method=c("knnImpute","center","scale"))
preProcValues1
# Otras opciones: range , bagImpute, medianImpute

datos_transformado1 <- predict(preProcValues1, datosD)

# Graficar la cantidad de valores perdidos en las 
# variables categoricas
graf_perdidos2 <- aggr(datos_transformado1,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

# Imputacion de datos categoricos
table(datos_transformado1$Gender)
table(datos_transformado1$Gender,useNA="always")

# Imputar valores missing usando el algoritmo Random Forest
library(missForest)
set.seed(123)
impu_cate          <- missForest(datos_transformado1)
datos_transformado1 <- impu_cate$ximp

# Verificando la cantidad de valores perdidos
plot_missing(datos_transformado1)

#----------------------------------------------------------------
# Opcion 2 - No sofisticada o univariada
# Utilizamos la libreria Mlr para esto.
library(mlr)

datosD_Imp <- mlr::impute(datosD, classes = list(factor = imputeMode(),  # Cualitativas por moda
                                                     integer = imputeMedian(), # Cuantitativas por media
                                                     numeric = imputeMedian()),
                            dummy.classes = c("integer","factor"), dummy.type = "numeric")

datosD_Imp=datosD_Imp$data[,1:min(dim(datosD))]

summary(datosD_Imp)

table(datosD_Imp$Loan_Status)
str(datosD_Imp)

#----------------------------------------------------------------
# Opcion : Ingenieria de datos
# Usando el paquete dummies
library(dummies)
# Dataframe con dummies e inputaciÛn sofisticada, numÈrica con KNN + Escalamiento
# y cualitativas, inputacion con RandomForest
datos_transformado2 <- dummy.data.frame(datos_transformado1,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))
# Dataframe con dummies e inputaciÛn no sofisticada, numÈrica con Mediana
# y moda para las variables cualitativas
datos_transformado3 <- dummy.data.frame(datosD_Imp,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))

# Verificando la estructura del archivo pre-procesado
str(datos_transformado2)
str(datos_transformado3)

#####################################
# 3. Seleccion de Drivers ###########
#####################################
# Seleccion de variables
# Utilizando Boruta

Bor.hvo<-Boruta(Loan_Status~.,data=datos_transformado2,doTrace=2);
windows()
plot(Bor.hvo,las=3)

#####################################
### 4. Particion Muestral ###########
#####################################

## Particionando la Data
table(datosD_Imp$Loan_Status)
table(datos_transformado2$Loan_Status)
set.seed(1234) # Opcional, si queremos que a todos nos salga igual

library(caret)
sample <- createDataPartition(datos_transformado2$Loan_Status, 
                              p = .70,list = FALSE,times = 1)

data.train <- datos_transformado2[ sample,] # Dataset de Entrenamiento
data.prueba <- datos_transformado2[-sample,] # Dataset de Validacion


#####################################
# 5. Modelamiento Predictivo ########
#####################################

##########################
###### Random Forest #######

set.seed(1234)
library(randomForest)
modelo_rf <- randomForest(Loan_Status~., # Y ~ X
                          data = data.train,   # Datos a entrenar 
                          ntree=200,           # Numero de arboles
                          mtry = 4,            # Cantidad de variables
                          # Raiz2 Total de variables o 40%-60% total variables.
                          importance = TRUE,   # Determina la importancia de las variables
                          replace=T) 
#----------------------------------
# OOB error (out of bag error)
#
print(modelo_rf)

# Graficar Error del Modelo
#
# En este gr?fico se muestra un modelo que intenta predecir 
# la variable churn={FUGA,ACTUAL}. 
# La linea negra representa el OOB, 
# la linea roja es el error al intentar predecir churn={ACTUAL}, 
# la linea verde es el error en la prediccion churn={FUGA}. 
# La linea negra siempre ser? el OOB, y las siguientes lineas
# se pueden identificar con la matriz de confusi?n 
# usando print(MODELO.RF) 

plot(modelo_rf)

#-----------------------------------------------------------------------------
# Importancia de las variables
# La tabla MeanDecreaseAccuracy representa en cu?nto removiendo 
# dicha variable se reduce la precision del modelo.
# Un valor mas alto de MeanDecreaseAccuracy o 
# del MeanDecreaseGiniScore, implica una mayor importancia 
# de la variable en el modelo.

varImpPlot(modelo_rf)
modelo_rf$importance


# Predecimos sobre la datatest
proba_rf <- predict(modelo_rf, # Modelo entrenado de RF
                    data.prueba,
                    type="prob")
head(proba_rf)
proba_rf <- proba_rf[,2]
# Curva ROC
library(pROC)
AUC <- roc(data.prueba$Loan_Status, proba_rf) 
auc_rf=AUC$auc
auc_rf
# Indice de gini
gini_rf <- 2*(AUC$auc) -1
gini_rf

# Calcular los valores predichos
PRED_rf <-predict(modelo_rf,data.prueba,type="class")

# Calcular la matriz de confusion
tabla=confusionMatrix(PRED_rf,data.prueba$Loan_Status,
                      positive = "1")
tabla
table(datos_transformado2$Loan_Status)
prop.table(table(datos_transformado2$Loan_Status))
##########################
###### AdaBoosting #######

library(adabag)
set.seed(1234)
# 'Freund', 'Breiman', 'Zhu' 
modelo_boosting<-boosting(Loan_Status~., # Y~X
                   data = data.train,    # Data train
                   coeflearn='Freund', # Learning rate
                   boos=TRUE, 
                   mfinal=50)          # Numero de arboles
modelo_boosting$importance
Importancia_Varia <- data.frame(modelo_boosting$importance)
write.csv(Importancia_Varia,"Importancia_Variables.csv")

# Predecimos
proba_boosting<-predict(modelo_boosting,data.prueba)
prob_boosting <- proba_boosting$prob[,2]
prob_boosting
# Curva ROC
AUC <- roc(data.prueba$Loan_Status, prob_boosting) 
auc_boosting=AUC$auc
auc_boosting
# Indice de gini
gini_boosting <- 2*(AUC$auc) -1
# Indice de discriminancia de las clases

# Calcular los valores predichos
PRED_boosting <-proba_boosting$class
PRED_boosting <- as.factor(PRED_boosting) # Cambio el tipo dato factor


# Calcular la matriz de confusion
tabla=confusionMatrix(PRED_boosting,data.prueba$Loan_Status,
                      positive = "1")
tabla

##########################
###### XGBoost ###########

library(xgboost)

# Separar covariables y target
Mtrain_XGB <- model.matrix(~ ., data=data.train[,c(1:21)]) # X
Ytrain <- as.vector(data.train$Loan_Status)                # Y

# Construimos una matriz Xgboost
dtrain <- xgb.DMatrix(Mtrain_XGB, label = Ytrain)

# Lo mismo que le hago al train, le hago al test
Mtest_XGB <- model.matrix(~ ., data=data.prueba[,c(1:21)])
Ytest <- as.vector(data.prueba$Loan_Status)
dtest <- xgb.DMatrix(Mtest_XGB, label = Ytest)

#Hacemos nuestra lista de particiones de datos
watchlist <- list(train = dtrain, test = dtest)

# Grilla N¬∞ 01
param <- list(max.depth = 1, # Profundidad arboles
              eta=0.00001,     # Ratio de aprendizaje
              silent = 0,    # Mostrar resultados
              alpha=0.8,       # L1
              #gamma=1,       # L2  
              objective="reg:logistic", # El tipo de modelo
              eval_metric="auc")

# Grilla N¬∞ 02
param <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eta=0.00001,
               alpha=0.8,
               gamma=0.7, 
               max_depth=1, 
               #min_child_weight=1, 
               subsample=0.5, 
               colsample_bytree=0.5,
               eval_metric = "auc")
# Entrenamiento
xgb_fit <- xgb.train(param, 
                     dtrain, 
                     nround = 1000, 
                     watchlist,verbose = 1,
                     early_stopping_rounds = 15)

# Importancia de las variables
importance_xgb <- xgb.importance(model = xgb_fit)
importance_xgb


# Prediccion
pred_xgb_t <- predict(xgb_fit,dtest,type="response")

# Convertimos la clase a probabilidad

clas_Xgb <- ifelse(pred_xgb_t<0.50,'0','1')
clas_Xgb <- as.factor(clas_Xgb)
# Vemos la matriz de confusion

matrizXGB <- caret::confusionMatrix(clas_Xgb,data.prueba$Loan_Status,positive='1')
matrizXGB

# Guardamos el modelo de ML
saveRDS(xgb_fit,"ModeloXG.rds")

#################################################################
##### (2) DESPLIEGUE DE ALGORITMOS DE MACHINE LEARNING ##########
#################################################################

# Leemos el dataset de despliegue de modelos #
DespliegueD<-read.csv("test.csv",na.strings = c(""," ",NA))  # leer la data de Validacion 

DespliegueD$Loan_ID <- NULL

#  Recodificacion de Variables para el Train
#  Recodificando Dependents
DespliegueD$Dependents=ifelse(DespliegueD$Dependents=="3+",3,
                         ifelse(DespliegueD$Dependents=="0",0,
                                ifelse(DespliegueD$Dependents=="1",1,
                                       ifelse(DespliegueD$Dependents=="2",2,
                                              DespliegueD$Dependents))))
DespliegueD$Dependents=as.factor(DespliegueD$Dependents)

#  Recodificando Gender
DespliegueD$Gender=ifelse(DespliegueD$Gender=="Male",1,0)
DespliegueD$Gender=as.factor(DespliegueD$Gender)


#  Recodificando Married
DespliegueD$Married=ifelse(DespliegueD$Married=="Yes",1,0)
DespliegueD$Married=as.factor(DespliegueD$Married)

#  Recodificando Education
DespliegueD$Education=ifelse(DespliegueD$Education=="Graduate",1,0)
DespliegueD$Education=as.factor(DespliegueD$Education)

#  Recodificando Self Employed
DespliegueD$Self_Employed=ifelse(DespliegueD$Self_Employed=="Yes",1,0)
DespliegueD$Self_Employed=as.factor(DespliegueD$Self_Employed)

#  Recodificando Property Area
DespliegueD$Property_Area=ifelse(DespliegueD$Property_Area=="Rural",0,
                            ifelse(DespliegueD$Property_Area=="Semiurban",1,
                                   2))
DespliegueD$Property_Area=as.factor(DespliegueD$Property_Area)

#  Recodificando Credit History
DespliegueD$Credit_History=as.factor(DespliegueD$Credit_History)

#  Imputacion de Datos

library(caret)
set.seed(123)
preProcValues1 <- caret::preProcess(DespliegueD,
                                    method=c("knnImpute","center","scale"))
DespliegueD2 <- predict(preProcValues1, DespliegueD)

library(missForest)
set.seed(123)
impu_cate          <- missForest(DespliegueD2)
DespliegueD3 <- impu_cate$ximp

library(dummies)
DespliegueD4 <- dummy.data.frame(DespliegueD3,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))

# library(mlr)
# datosDes_Imp<- mlr::impute(DespliegueD, classes = list(factor = imputeMode(),  # Cualitativas por moda
#                                                  integer = imputeMedian(), # Cuantitativas por media
#                                                  numeric = imputeMedian()),
#                           dummy.classes = c("integer","factor"), dummy.type = "numeric")


# datosDes_Imp=datosDes_Imp$data[,1:min(dim(DespliegueD4))]
# 
# summary(datosDes_Imp)
# 

# Prediccion con XGBoost

library(xgboost)
Mdespliegue_XGB <- model.matrix(~ ., data=DespliegueD4)

pred_xgb_d <- predict(xgb_fit,Mdespliegue_XGB,type="response")

# Convertimos la prediccion a clase para mandarla al submmit

submmit_Xgb <- ifelse(pred_xgb_d<0.50,'N','Y')

# volver a cargar la base para poder crear el dataframe:
submmit <- data.frame(Loan_ID=DespliegueD$Loan_ID,
                     Loan_Status=submmit_Xgb)

# Lo mandamos a campanas o Analytics Vidyha
write.csv(submmit,"Sol_Xgboost001.csv",row.names = F)
# FIN!

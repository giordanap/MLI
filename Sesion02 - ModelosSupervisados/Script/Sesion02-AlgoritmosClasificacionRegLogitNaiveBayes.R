#########################################################################
#########------- Machine Learning Inmersion ------------#################
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com
# Tema: Regresion Logistica - NaiveBayes - Knn
# version: 1.0
#########################################################################


#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls()) ; dev.off()

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

#---------------------------------------------------------
# Paquetes 
library(MASS)    # Modelos Logisticos, Intervalos confianza
library(pROC)    # Curva Roc-Auc
library(foreign) # Lectura de archivos en otros formatos: SPSS
library(gmodels) # Graficos y modelos
library(InformationValue) # Importancia de variables, WOESS
library(caTools) # Herramientas para indicadores de validacion de modelos
library(caret)   # Libreria de Machine Learning
library(ggplot2) # Graficas
library(MLmetrics) # Herramientas para indicadores de validacion de modelos
library(ISLR)      # Libreria del libro de los creadores de Rf

##############################
#  CASO 2. Fuga de Clientes  #
##############################

# El objetivo es predecir clientes propensos a desafilarse 
# (tambien llamado churn o attrition) en una empresa de 
# telecomunicaciones.
# Se cuenta con una data de 1345 clientes de una empresa de 
# telecomunicaciones donde algunos siguen siendo clientes (Actual) 
# y otros han fugado de manera voluntaria (Fuga).
#
# Variable Dependiente:  
#           CHURN   (0=Cliente Actual, 1=Fuga Voluntaria)
# Variables Independentes:
#           EDAD    (Edad del cliente en anos)  
#           SEXO    (Sexo del cliente, 1=Fememino 2=Masculino)
#           CIVIL   (Estado civil del cliente, 1=Casado 2=Soltero) 
#           HIJOS   (Numero de hijos del cliente)
#           INGRESO (Ingresos anuales del cliente)
#           AUTO    (Si el cliente es dueno de un auto, 1=Si 2=No)  
# Variable de identificacion: 
#           ID      (Codigo del cliente)

#----------------------------------------------------------
# Preparacion de los datos
library(foreign)
datos.r <-read.spss("Churn-arboles.sav",
                  use.value.labels=TRUE, 
                  to.data.frame=TRUE)
str(datos.r)
View(datos.r)

# No considerar la variable de identificacion ID
datos.r$ID <- NULL
str(datos.r)

# Etiquetando las opciones de las variables categoricas
levels(datos.r$CHURN)
levels(datos.r$SEXO)  <- c("Fem","Masc")
levels(datos.r$CIVIL) <- c("Casado","Soltero")
levels(datos.r$AUTO)  <- c("Si","No")
levels(datos.r$CHURN) <- c("No Fuga","Fuga")

str(datos.r)

# Direccionar a r, al dataset
attach(datos.r)

# Para cambiar la categoria de referencia
datos.r$SEXO = relevel(datos.r$SEXO,ref="Masc") # Cambio de referencia
contrasts(datos.r$SEXO)

#--------------------------------------------------------------
# PARTE 1: MODELAMIENTO MEDIANTE ALGORITMOS DE MACHINE LEARNING
#--------------------------------------------------------------

#-------------------------------------------------------------------
# Seleccion de muestra de entrenamiento (70%) y de prueba (30%)
library(caret)
set.seed(123) 
index <- createDataPartition(datos.r$CHURN, p=0.7, list=FALSE)
training <- datos.r[ index, ] # Entrenamiento del modelo
testing <-  datos.r[-index, ] # Validacion o test del modelo

# Verificando la estructura de los datos particionados
prop.table(table(datos.r$CHURN)) # Distribucion de y en el total
prop.table(table(training$CHURN))
prop.table(table(testing$CHURN))

###################################################
######## REGRESION LOGISTICA BINARIA ##############
###################################################

options(scipen=999)

modelo_churn <- glm(CHURN ~ . ,  # Todas las variables
                    family=binomial,
                    data=training) # Data de entrenammiento

summary(modelo_churn)
coef(modelo_churn)

#--------------------------------------------------------------------------
# Cociente de ventajas (Odd Ratio)
exp(coef(modelo_churn))

# (Intercept)       EDAD    SEXOFem  CIVILSoltero        HIJOS      INGRESO 
#   0.1613089  1.0081315  12.3560732    1.0774856    0.9425963    0.9999911 
#      AUTONo 
#   0.9016556 

# Para el caso de SEXO, el valor estimado 12.356 significa que, 
# manteniendo constantes el resto de las variables, 
# que las personas del g?nero FEMENINO tienen 12.356 veces m?s ventaja 
# de FUGAR que los sujetos que son del g?nero MASCULINO.

# Para el caso de la EDAD, ante un incremento en una unidad de medida de 
# la EDAD (un a?o), provocar? un incremento multiplicativo por un factor 
# de 1.008 de la ventaja de FUGA 

cbind(Coeficientes=modelo_churn$coef,ExpB=exp(modelo_churn$coef))

#------------------------------------------------------------------------
# Cociente de ventajas e Intervalo de Confianza al 95% 
library(MASS)
exp(cbind(OR = coef(modelo_churn),confint.default(modelo_churn)))

#----------------------------------------------------------
# Importancia de las variables, Feature Selection*
varImp(modelo_churn)

#----------------------------------------------------------
# Seleccion de Variables  
library(MASS)
step <- stepAIC(modelo_churn,direction="backward", trace=FALSE)
step$anova

#----------------------------------------------------------
# Modelo 2 con las variables mas importantes
modelo_churn2 <- glm(CHURN ~ EDAD + SEXO + INGRESO, 
                     family=binomial,
                     data=training)

summary(modelo_churn2)
coef(modelo_churn2)

library(MASS)
exp(cbind(OR = coef(modelo_churn2),confint.default(modelo_churn2)))

# Como tenemos un modelo parsimonioso creado , lo guardamos.
saveRDS(modelo_churn2,"Rg_Logistica.rds")
#----------------------------------------------------------
######### Despliegue o Score de Nuevos Indiviuos ##########
#----------------------------------------------------------
# Prediccion para nuevos individuos  
nuevo1 <- data.frame(EDAD=57, SEXO="Fem",INGRESO=27535.30)
predict(modelo_churn2,nuevo1,type="response")

nuevo2 <- data.frame(EDAD=80, SEXO="Fem",INGRESO=12535.50)
predict(modelo_churn2,nuevo2,type="response")


############################################
#  INDICADORES PARA EVALUACION DE MODELOS  #
############################################

#-----------------------------------------------------------------
# Para la evaluacion se usara el modelo_churn2 obtenido con la 
# muestra training y se validara en la muestra testing

# Prediciendo la probabilidad
proba.pred <- predict(modelo_churn2,testing,type="response")
head(proba.pred)

# Prediciendo la clase (con punto de corte = 0.5)
clase.pred <- ifelse(proba.pred >= 0.5, 1, 0)

head(clase.pred)

str(clase.pred)

# Convirtiendo a factor
clase.pred <- as.factor(clase.pred)          

levels(clase.pred) <- c("No Fuga","Fuga")

str(clase.pred)

head(cbind(testing,proba.pred,clase.pred),8)

write.csv(cbind(testing,proba.pred,clase.pred),
          "Testing con clase y proba predicha-Logistica.csv")

# Graficando la probabilidad predicha y la clase real
ggplot(testing, aes(x = proba.pred, fill = CHURN)) + 
  geom_histogram(alpha = 0.25)

#############################
# 1. Tabla de clasificacion #
#############################

#---------------------------------------------
# Calcular el % de acierto (accuracy)
accuracy <- mean(clase.pred==testing$CHURN)
accuracy

#---------------------------------------------
# Calcular el error de mala clasificacion
error <- mean(clase.pred!=testing$CHURN)
error

library(gmodels)
CrossTable(testing$CHURN,clase.pred,
           prop.t=FALSE, prop.c=FALSE,prop.chisq=FALSE)

# Usando el paquete caret
library(caret)
caret::confusionMatrix(clase.pred,testing$CHURN,positive="Fuga")

############################
# 2. Estadistico de Kappa  #
############################

# Tabla de Clasificaci?n
addmargins(table(Real=testing$CHURN,Clase_Predicha=clase.pred))

#           Clase_Predicha
# Real     Actual Fuga Sum
# Actual      168   81 249
# Fuga         25  128 153
# Sum         193  209 402

# pr_o es el Accuracy Observado o la Exactitud Observada del modelo
pr_o <- (168+128)/402 ; pr_o

# pr_e es el Accuracy Esperado o la Exactitud Esperada del modelo
pr_e <- (249/402)*(193/402) + (153/402)*(209/402) ; pr_e

# Estad?stico de Kappa
k <- (pr_o - pr_e)/(1 - pr_e) ; k

#####################################
# 3. Estadistico Kolgomorov-Smirnov #
#####################################

#---------------------------------
# Calculando el estadistico KS
library(InformationValue)

ks_stat(testing$CHURN,proba.pred, returnKSTable = T)
ks_stat(testing$CHURN,proba.pred)

# Graficando el estadistico KS 
ks_plot(testing$CHURN,proba.pred)

#####################################
# 4. Curva ROC y area bajo la curva #
#####################################

#----------------------------------------------
# Usando el paquete pROC
library(pROC)

# Area bajo la curva
roc <- roc(testing$CHURN,proba.pred)
roc$auc

#---------------------------------------------------
# Curva ROC usando el paquete caTools
library(caTools)
AUC <- colAUC(proba.pred,testing$CHURN, plotROC = TRUE)
abline(0, 1,col="red") 

AUC  # Devuelve el area bajo la curva

puntos.corte <- data.frame(prob=roc$thresholds,
                           sen=roc$sensitivities,
                           esp=roc$specificities)
head(puntos.corte)

# Punto de corte optimo (mayor sensibilidad y especificidad) usando pROC
coords(roc, "best",ret=c("threshold","specificity", "sensitivity","accuracy"))
coords(roc, "best")

plot(roc,print.thres=T)

# Graficando la Sensibilidad y Especificidad
ggplot(puntos.corte, aes(x=prob)) + 
  geom_line(aes(y=sen, colour="Sensibilidad")) +
  geom_line(aes(y=esp, colour="Especificidad")) + 
  labs(title ="Sensibilidad vs Especificidad", 
       x="Probabilidad") +
  scale_color_discrete(name="Indicador") +
  geom_vline(aes(xintercept=0.5507937),
             color="black", linetype="dashed", size=0.5) + 
  theme_replace() 


##########################
# 5. Coeficiente de Gini #
##########################

gini <-  2*AUC -1 ; gini

################
# 6. Log Loss  #
################

# Transformar la variable CHURN a numerica
real <- as.numeric(testing$CHURN)
head(real)
# [1] 1 1 2 1 1 2

# Recodificar los 1 y 2 como 0 y 1 respectivamente
real <- ifelse(real==2,1,0)

library(MLmetrics)
LogLoss(proba.pred,real)

##########################################################
### REGRESION LOGISTICA CON CARET Y VALIDACION CRUZADA ###
##########################################################

# Relacion de parametros a ajustar de un modelo
modelLookup(model='glm')

# Aplicando el modelo con Validacion Cruzada 
ctrl <- trainControl(method="cv",number=10)

set.seed(123)
modelo_log <- train(CHURN ~ ., 
                    data = training, 
                    method = "glm", family="binomial", 
                    trControl = ctrl, 
                    tuneLength = 5,
                    metric="Accuracy")
modelo_log

summary(modelo_log)

plot(modelo_log)

varImp(modelo_log)


#######################################
#### ALGORITMO DE NAIVE BAYES #########
#######################################
library(e1071)
modelo_naiveB=naiveBayes(CHURN~.,
                   data=training)
pred_bayes<- predict(modelo_naiveB,testing)
pred_bayes
tabla=confusionMatrix(pred_bayes,
                      testing$CHURN,positive = "Fuga")

# Tabla de matriz de clasificacion
tabla

#######################################
#### ALGORITMO DE KNN #################
#######################################
# Para algoritmos como knn, debemos tener los datos en formato numerico,
# ademas de estandarizar o normalizar los valores. 

# Para los datos de training
training$SEXO <- ifelse(training$SEXO=="Fem","0","1")
training$SEXO <- as.factor(training$SEXO)

training$CIVIL <- ifelse(training$CIVIL=="Casado","0","1")
training$CIVIL <- as.factor(training$CIVIL)

training$AUTO <- ifelse(training$AUTO=="No","0","1")
training$AUTO <- as.factor(training$AUTO)

# Para los datos de testing
testing$SEXO <- ifelse(testing$SEXO=="Fem","0","1")
testing$SEXO <- as.factor(testing$SEXO)

testing$CIVIL <- ifelse(testing$CIVIL=="Casado","0","1")
testing$CIVIL <- as.factor(testing$CIVIL)

testing$AUTO <- ifelse(testing$AUTO=="No","0","1")
testing$AUTO <- as.factor(testing$AUTO)


library(class)
pred_knn <- knn(training[,1:6],testing[,1:6],
                cl=training$CHURN,k=10)
pred_knn

# Calcular la matriz de confusion
tabla=confusionMatrix(pred_knn,
                      testing$CHURN,positive = "Fuga")
tabla


#------------------------------------------------------
# PARTE 2: DESPLIEGUE Y PRODUCTIVO DE MODELOS
#------------------------------------------------------

# Deseamos replicar o implementar el modelo.
# Leemos el modelo predictivo.
RegresionLog <- readRDS("Rg_Logistica.rds")

# Leemos el dataset de nuevos leads
library(foreign)
datos_n <-read.spss("Churn-nuevos-arboles.sav",
                    use.value.labels=TRUE, 
                    to.data.frame=TRUE)

# Decodificacion

levels(datos_n$SEXO)  <- c("Fem","Masc")
levels(datos_n$CIVIL) <- c("Casado","Soltero")
levels(datos_n$AUTO)  <- c("Si","No")


# Scorear o puntuar nuevos registros
base_gestionNov19 <- predict(RegresionLog,datos_n,type = "response")
# Convertir a prediccion
Score_gestionNov19 <- ifelse(base_gestionNov19<=0.50,"No Fuga","Fuga")
# Lo mandamos a gestionar a distintas areas
BaseGestNov19 <- data.frame(
        DNI=datos_n$ID,
        Score_Predict=Score_gestionNov19)

# Exportar el objeto
write.csv(BaseGestNov19,"BaseCampanas-Nov19.csv",row.names = F)

# FIN !!
# Correo : andre.chavez@urp.edu.pe

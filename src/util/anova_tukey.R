# Se precisar carregar pacotes adicionais, siga os exemplos abaixo 
# install.packages("psych")
# install.packages("ggplot2")
# library(psych)

library("ggplot2")

# Ler os dados de um arquivo (interagindo com o usuario)
dados <- read.table(file.choose(),header=TRUE)

# Mostra boxplots das técnicas, lado a lado, em relação ao desempenho
dados$tecnica <- as.factor(dados$tecnica)
bp <- ggplot(dados, aes(x=tecnica, y=desempenho,fill=tecnica)) + 
  geom_boxplot()+
  labs(title="Boxplot da Medida-F por Técnica de Extração de Atributos",x="Técnica", y = "Medida-F")
bp + theme_classic()


# Mostra boxplots das classes, lado a lado, em relação ao desempenho
dados$classe <- as.factor(dados$classe)
bp <- ggplot(dados, aes(x=classe, y=desempenho,fill=classe)) + 
  geom_boxplot()+
  labs(title="Boxplot da Medida-F por Espécie de Peixe",x="Espécie de Peixe", y = "Medida-F")
bp + theme_classic()


# Cria a tabela ANOVA usando as classes como blocos 
# Na terminologia mais comum:
# classe= bloco
# tecnica=tratamento
dados.anova <- aov(dados$desempenho ~ dados$tecnica + dados$classe)

# Mostra a tabela ANOVA
summary(dados.anova)

# Realiza e mostra os resultados de um pós-teste usando Tukey
# Primeiro para as técnicas
tukey <- TukeyHSD(dados.anova,'dados$tecnica',conf.level=0.95)
tukey

par(mar=c(4,10,4,4))
plot(tukey , las=1 , col="brown" )

# Depois para as classes (espécies)
tukey <- TukeyHSD(dados.anova,'dados$classe',conf.level=0.95)
tukey

par(mar=c(4,10,4,4))
plot(tukey , las=1 , col="brown" )


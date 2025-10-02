# Análise do Desastre do Titanic

Repositório criado para a [competição do Kaggle sobre o desastre do Titanic](https://www.kaggle.com/competitions/titanic) e baseado nas aulas do canal [Hashtag Programação](https://www.youtube.com/@HashtagProgramacao) no YouTube.

Os resultados desta competição utilizando diversas ferramentas de Machine Learning (ML) que calculam a taxa de acerto dos modelos foram separados em 5 partes, sendo os seguintes resultados:
1. 66.75%
2. 76.56%
3. 76.32%
4. 74.88%
5. 77.99%

## [Etapa 1: Primeiro Modelo](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte1.ipynb)

* Nesta primeira etapa, realizei um estudos iniciais sobre análises exploratórias com **ydata-profiling**, tratamento e limpeza de dados com **Pandas** e modelos de Machine Learning com **scikit-learn**.
  - O processo de análise de dados foi divido em duas etapas. Inicialmente, utilizou-se o **ydata-profiling** para uma análise exploratória superficial, para entender os dados, principalmente quais colunas do DataFrame atribuído ao dataset estavam vazias e quais eram suas respectivas classes.
  - Em seguida, a limpeza e o tratamento dos dados foi realizado utilizando puramente a biblioteca do **Pandas**, se atentando principalmente com dados nulos ou faltantes, retirando colunas com alta cardinalidade[1] e atribuindo modelagens estatísticas para dados faltantes, como a média para colunas numéricas e a moda para colunas categóricas. Além disso, a base numérica foi separada da base categórica para realizar a primeira previsão com o modelo de ML.
  - Por fim, utilizou-se três diferentes modelos de ML da biblioteca **scikit-learn**, entendo a funcionalidade de cada modelo. Um estudo prévio à este trabalho foi feito para entender os conceitos por trás de cada modelo aqui utilizado, o qual pode ser encontrado no repositório [Modelos de Machine Learning](https://github.com/Marinopl/Modelos-de-Machine-Learning). Aqui, utilizamos três modelos: Árvore de Decisão por Classificação, KNeighborsClassifier e Regressão Logística por classificação. Para todas os modelos, realizou-se o estudo de acertos por duas métricas: **acurácia** e a **matriz de confusão**, que podem ser encontrados neste [repositório](https://github.com/Marinopl/Modelos-de-Machine-Learning/tree/main/Avaliando%20Modelo%20de%20Classifica%C3%A7%C3%A3o).
    - Como a competição do Kaggle considerou a acurácia como métrica para pontuação de previsões, escolhemos o modelo com maior acurácia para realizar as previsões: o modelo de Regressão Logística.
   
* O score no Kaggle para esta primeira etapa foi uma taxa de acerto de 66.75%

[1] Cardinalidade considera a quantidade de valores distintos em uma coluna atribuida a uma feature. Com valores muito dispersos, o modelo não consegue realizar uma boa previsão. Normalmente, se a feature não influencia muito na previsão, esta coluna com alta cardinalidade pode ser excluída.


## [Etapa 2: Tratamento de variáveis de texto](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte2.ipynb)

* Na última etapa, utilizamos apenas variáveis numéricas para treinar nossos modelos. A pergunta é: existem variáveis categóricas no nosso modelo que podem ajudar na previsão?
  - Duas colunas chamam a nossa atenção: Genêro (Sex) e Embarque (Embarked)
     - A coluna Sex possui dois genêros, femino e masculino, a ideia é criar uma nova coluna numérica atribuindo valores booleanos, 1 para masculino e 0 para feminino.
     - A coluna Embarked possui três categorias, o que torna a atribuição de valores booleanos não verdadeira, por isso, a solução utilizada aqui foi a ferramente **OneHotEncoder**. Esta ferramenta atribui valores binários para as três categorias que aparecem em Embarked. Ao ser aplicada, ela cria três novas colunas, atribuindo 1 para a categoria existente na coluna inicial.
     - Por fim, após tratar estas duas colunas, retira-se as colunas categóricas Sex e Embarked.
   
    - Em seguida, utiliza-se a nova base com as colunas de texto tratadas para treinar os modelos de ML, utilizando os mesmos modelos da etapa anterior.
       - Aqui, o modelo com maior acurácia foi o de Regressão Logística
     
* O score no Kaggle para esta etapa foi uma taxa de acerto de 76.56%

## [Etapa 3: Analisando escalas e novas features](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte3.ipynb)

* Nesta etapa, utilizamos a base de dados gerada na [Etapa 2](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte2.ipynb), já que ela obteve uma maior pontuação no Kaggle. Entretanto, aqui foi necessário apronfundar o entendimento sobre a escala dos dados[2], em especial, colunas como Idade (Age) e Preço da Passagem (Fare) possuem valores muito distintos, tornando o dataset mais difícil de ser modelado.
  - Solução: aplicou-se uma normalização com RobustScaler, já que é um método mais robusto à outliers, já que a mediana não é influenciada por valores extremos.
 
* Em seguida, esta nova base de dados, após a normalização, foi utilizada para treinar nossos modelos de ML e prever os resultados para a nossa base de teste. Aqui, o modelo com maior acurácia foi o de Regressão Logística.

* O score no Kaggle para esta etapa foi uma taxa de acerto de 76.32%.

* Além desses passos, aproveitei para analisar outras colunas e entender a correlação entre elas e o restante da base de dados. Foi uma análise construtiva para entender a história por trás dos dados observados e como eles se correlacionavam por trás dos números. Em especial, descobriu-se que pessoas que viajam sozinhas no Titanic, tiveram uma taxa de sobrevivência menor do que aqueles que viajavam com mais de uma pessoa.
  - Com esta nova base de dados, também realizei a métrica de acurácia, mas não utilizei este novo modelo na previsão do Kaggle, já que a acurácia dos modelos cairam em relação à etapa de normalização.


[2] Escala dos dados remete-se a comparação de uma feature em relação as outras em respeito aos valores númericos em um mesmo intervalo. Muitos algoritmos de ML (como Regressão Logística, KNN, SVM e Neural Networking), isto é, modelos que utilizam a distância euclidiana como métrica de classificação, são muito sensíveis à escalabilidade dos dados:
  - Features com intervalos numéricos maiores dominam as métricas de distância;
  - O processo de otimização pode ficar mais difícil se houver pesos (bias) para a feature fora de escala.


## [Etapa 4: Utilizando novos modelos de Machine Learning](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte4.ipynb)

* Nesta etapa, utilizamos outros modelos de Machine Learning da biblioteca scikit-learn, como Random Foreste e MLPClassifier (o qual é baseado em Neural Networking).

* De início, utilizamos uma base de dados modelada pela [Etapa 2](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte2.ipynb) e pela [Etapa 3](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte3.ipynb), realizando o treinamento dos novos modelos e de Regressão Logística, o qual foi escolhido dentre os três primeiros pois obteve a maior taxa de acerto.
  - Dentre os modelos (Regressão Logística, Random Forest, MLPClassifier), o que obteve maior acurácia foi o de Redes Neurais (MLPClassifier), logo foi utilizado para realizar a previsão da base de testes.
 
* O score no Kaggle para esta etapa foi uma taxa de acerto de 74.88%.

* Houve uma leve piora em relação à útlima etapa, o que pode ser entendido como um possível *overfitting* do modelo.

## [Etapa 5: Encontrando os melhores parâmetros com GridSearchCV](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte5.ipynb)

* Aqui utilizamos a base de dados da [Etapa 3](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte3.ipynb), onde também foi aplicada aos novos modelos escolhidos na [Etapa 4](https://github.com/Marinopl/Desastre-Titanic/blob/main/An%C3%A1lise%20do%20Titanic/Parte4.ipynb), Random Forest e MLPClassifier, assim como o de Regressão Logística.
  - Buscamos entender quais são os melhores parâmetros para o treinamento dos modelos com o GridSearchCV, juntamente ao KFold, ambas ferramentas da biblioteca do scikit-learn.
  - Após encontrar os melhores parâmetros, realizou-se a previsão da base de teste com os três modelos, obtendo um melhor *score* para o modelo de Random Forest e, após avaliação dos modelos pela métrica da acurácia, obteve-se uma acurácia igual para os modelos de Random Forest e MLPClassifier, maior do que a acurácia do modelo de Regressão Logística.

* O score obtido no Kaggle para os dois modelos, Random Forest e MLPClassifier, foram iguais com uma taxa de acerto de 77.99%.

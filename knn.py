# Classificador de câncer de mama
#Neste projeto, usaremos um classificador K-Nearest Neighbor para prever se um paciente tem câncer de mama.
### Carregando o conjunto de dados
#Vamos obter os dados de cancêr de mama do próprio `sklearn` importando a função `load_breast_cancer` do `sklearn.datasets` 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#**1.** Depois de importar o conjunto de dados, vamos carregar os dados em uma variável chamada `dados_cancer_mama`. Faça isso configurando `dados_cancer_mama` igual à função `load_breast_cancer()`.
dados_cancer_mama = load_breast_cancer()

#**2.** Antes de começarmos a criar nosso classificador, vamos dar uma olhada nos dados. Comece imprimindo `dados_cancer_mama.data[0]`. Esse é o primeiro ponto de dados em nosso conjunto. Mas o que todos esses números representam? Imprima também `dados_cancer_mama.feature_names`.
print(dados_cancer_mama.data[0])
print(dados_cancer_mama.feature_names)

#**3.** Agora temos uma noção de como são os dados, vamos verificar o que estamos tentando classificar? Vamos imprimir ambos `dados_cancer_mama.target` e `dados_cancer_mama.target_names`.
print(dados_cancer_mama.target)
print(dados_cancer_mama.target_names)

#O primeiro ponto de dados foi marcado como maligno ou benigno?

### Dividindo os dados em conjuntos de treinamento e teste
#**4.** Divida os dados em conjuntos de treinamento e teste usando o método `train_test_split()` do `sklearn`. Use um `test_size` de 0.2 e `random_state = 100`. Isso garantirá que toda vez que você executar seu código, os dados sejam divididos da mesma maneira.
x_train, x_test, y_train, y_test = train_test_split(dados_cancer_mama.data, dados_cancer_mama.target, test_size=0.2, random_state=100)

### Executando o KNN
#**5.** Agora que criamos conjuntos de treinamento e teste, podemos criar um `KNeighborsClassifier` e testar sua precisão. Comece importando `KNeighborsClassifier` de `sklearn.neighbors`
from sklearn.neighbors import KNeighborsClassifier

#**6.** Crie um `KNeighborsClassifier` onde n_neighbors = 3. Nomeie o classificador como `knn`
knn = KNeighborsClassifier(n_neighbors=3)

#**7.** Treine seu classificador usando a função `fit`. Esta função recebe dois parâmetros: o conjunto de treinamento e os rótulos de treinamento.
knn.fit(x_train, y_train)

#**8.** Agora que o classificador foi treinado, vamos descobrir o quão preciso ele é no conjunto de teste. Chame a função `score` do classificador. `score` recebe dois parâmetros: o conjunto de teste e os rótulos de teste. Imprima o resultado!
print(f'Precisão: {knn.score(x_test, y_test)}')

#**9.** O classificador se sai muito bem quando `k = 3`. Mas talvez haja um `k` melhor. Teste o classificador knn com valores de `k` de `1` até `100`.
#Qual `k` apresenta o melhor resultado?
scores = []
for k in range(1, 101):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train, y_train)
  scores.append(knn.score(x_test, y_test))

### Apresentando os resultados
#**10.** Agora temos a precisão para 100 `k`s diferentes. Em vez de apenas imprimir, vamos fazer um gráfico usando `matplotlib`
#O eixo x deve ser os valores `k` que testamos. Esta deve ser uma lista de números entre 1 e 100.
#O eixo y do nosso gráfico deve ser a precisão do conjunto de teste.

plt.plot(range(1, 101), scores)
plt.xlabel('k')
plt.ylabel('Precisão')
plt.show()
plt.clf()

#**11.** Imprima a matriz de confusão, utilizando os dados do conjunto de teste, do modelo com o `k` que obteve o maior `score` .
maior_score = scores.index(max(scores)) + 1

knn = KNeighborsClassifier(n_neighbors=maior_score)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Maligno', 'Benigno'])
disp.plot()
plt.show()
plt.clf()


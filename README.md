# Projeto-SIN393
Projeto de Processamento de Imagens e Classificação de Objetos destinado a disciplina de visão computacional SIN393 lecionada pelo professor João Fernando Mari.

Este projeto tem como objetivo realizar o processamento de imagens para segmentação de objetos, extração de características geométricas e treinamento de um modelo de machine learning para classificação. A seguir, explicamos cada parte do projeto e seu funcionamento.

Estrutura do Projeto
O projeto é composto por funções que manipulam imagens, extraem informações sobre os objetos presentes e treinam um modelo de machine learning utilizando essas informações. A estrutura básica do código é a seguinte:

Segmentação de Imagem
Cálculo de Características Geométricas
Pré-processamento para Machine Learning
Treinamento e Avaliação do Modelo
1. Segmentação de Imagem
Função: segment_image(image)
Essa função é responsável por segmentar objetos em uma imagem, removendo o fundo e destacando as formas dos objetos de interesse.

Entrada: A função recebe uma imagem (em formato RGB ou escala de cinza) como entrada.

Processo:

Conversão para Escala de Cinza: Caso a imagem não esteja em escala de cinza, ela é convertida.
Thresholding: A função aplica um limiar (thresholding) para separar os objetos do fundo, transformando a imagem em uma imagem binária (preto e branco).
Operações Morfológicas: São realizadas operações de fechamento para limpar a segmentação e remover pequenos ruídos ou lacunas.
Extração de Contornos: Os contornos dos objetos são extraídos para identificar a forma e a área dos objetos.
Criação da Máscara e Segmentação Final: A máscara gerada pelos contornos é aplicada à imagem original para segmentar e extrair os objetos de interesse.
Saída: A imagem segmentada, onde apenas os objetos de interesse são visíveis, com o fundo removido.

2. Cálculo de Características Geométricas
Função: calculate_features(image)
Essa função calcula diversas características geométricas dos objetos segmentados, as quais serão usadas para treinar um modelo de machine learning.

Entrada: A função recebe uma imagem binária (após segmentação).
Processo:
Área do Objeto: Calcula a área do objeto, que é a soma dos pixels dentro do contorno.
Eixos Maior e Menor da Elipse: Ajusta uma elipse ao contorno do objeto e calcula os eixos maior e menor, que indicam a forma do objeto.
Solidez: Calcula a razão entre a área do objeto e a área de seu casco convexo, indicando o grau de "compactação" do objeto.
Excentricidade: Calcula a excentricidade do objeto, que é uma medida do quanto o objeto se desvia de uma forma circular.
Saída: Um dicionário contendo as características calculadas: área, eixos maior e menor, solidez e excentricidade.
3. Cálculo dos Momentos de Hu
Função: calculate_hu_moments(image)
Os Momentos de Hu são invariantes geométricos que descrevem a forma dos objetos. Eles são usados para comparar formas de objetos em diferentes escalas, orientações e transladações.

Entrada: A função recebe uma imagem binária.
Processo:
Cálculo dos Momentos: São calculados os momentos de contorno da imagem, que fornecem uma descrição matemática da forma do objeto.
Normalização Logarítmica: Para estabilizar os valores, é aplicada uma transformação logarítmica nos Momentos de Hu.
Saída: Os Momentos de Hu da imagem, que são invariantes e podem ser usados como entradas para um modelo de machine learning.
4. Cálculo da Assinatura do Contorno
Função: calculate_contour_signature(image)
A assinatura do contorno de um objeto descreve a forma do contorno, baseado nas distâncias entre o centro de massa (centroide) e os pontos de contorno.

Entrada: A função recebe uma imagem binária (após segmentação).

Processo:

Extração do Contorno: Os contornos são extraídos usando cv2.findContours.
Cálculo do Centro de Massa: O centro de massa (centroide) do contorno é calculado.
Cálculo das Distâncias: As distâncias entre o centro de massa e os pontos do contorno são calculadas, formando a assinatura do contorno.
Saída: A assinatura do contorno, que descreve a forma do objeto em relação ao seu centro de massa.

5. Pré-processamento para Machine Learning
Após a extração das características geométricas e dos Momentos de Hu, é necessário preparar os dados para treinamento de um modelo de machine learning. Isso envolve:

Criação de um Conjunto de Dados: As características extraídas são armazenadas em uma matriz X (onde cada linha representa um objeto e cada coluna representa uma característica). Os rótulos dos objetos (por exemplo, categorias ou classes) são armazenados em um vetor y.
Divisão dos Dados: Os dados são divididos em conjuntos de treinamento e teste usando train_test_split do sklearn.
Normalização: A normalização das características é feita usando StandardScaler para garantir que todas as características tenham a mesma escala e o modelo de machine learning tenha um desempenho mais eficiente.
6. Treinamento e Avaliação do Modelo
Uma vez que os dados foram preparados, o modelo de machine learning pode ser treinado.

Exemplo de Treinamento com KNN:
Criação do Modelo: Utilizamos o classificador KNeighborsClassifier (KNN), que classifica os objetos com base na proximidade das suas características.
Treinamento do Modelo: O modelo é treinado com o conjunto de treinamento usando model.fit(X_train, y_train).
Avaliação do Modelo: Após o treinamento, o modelo é avaliado com o conjunto de teste. As previsões feitas pelo modelo são comparadas com os rótulos reais usando métricas como acurácia, precisão, recall e F1-Score.

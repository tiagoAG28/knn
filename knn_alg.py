import streamlit as st  # Importa o Streamlit para a interface web interativa
import numpy as np  # Importa o NumPy para operações numéricas
import matplotlib.pyplot as plt  # Importa o Matplotlib para criação de gráficos
import pandas as pd  # Importa o Pandas para manipulação e exibição de tabelas


# ==================== Classe KNN ====================
class KNNClassifier:
    """
    Classe que implementa o algoritmo KNN para classificação.

    O classificador armazena os dados de treino e permite a previsão das classes de novos pontos
    com base na votação dos k vizinhos mais próximos. Pode também normalizar os dados antes de calcular
    as distâncias, garantindo que todas as features tenham a mesma influência.
    """

    def __init__(self, k=3, normalize=True):
        """
        Inicializa a instância do classificador KNN.

        Parâmetros:
        -----------
        k : int
            Número de vizinhos a serem considerados na classificação.
        normalize : bool
            Se True, os dados serão normalizados (min-max scaling) antes da classificação.
        """
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.min = None  # Valores mínimos de cada feature para normalização
        self.max = None  # Valores máximos de cada feature para normalização

    def fit(self, X, y):
        """
        Treina o classificador armazenando os dados de treino.
        Se a normalização estiver ativada, os dados são transformados usando min-max scaling.

        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados de treino.
        y : array-like, shape = [n_samples]
            Rótulos correspondentes aos dados de treino.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.normalize:
            self.min = self.X_train.min(axis=0)
            self.max = self.X_train.max(axis=0)
            diff = self.max - self.min
            diff[diff == 0] = 1  # Evita divisão por zero para features constantes
            self.X_train = (self.X_train - self.min) / diff

    def predict(self, X):
        """
        Prevê os rótulos para os dados de teste com base nos dados de treino.

        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados para os quais se deseja prever os rótulos.

        Retorna:
        --------
        predictions : numpy array, shape = [n_samples]
            Vetor com os rótulos preditos para cada amostra.
        """
        X = np.array(X)
        if self.normalize:
            diff = self.max - self.min
            diff[diff == 0] = 1
            X = (X - self.min) / diff

        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Ordena os índices manualmente com base nas distâncias
            indices = list(range(len(distances)))
            indices.sort(key=lambda i: distances[i])
            k_indices = indices[: self.k]
            # Recolhe os rótulos dos k vizinhos
            k_nearest_labels = self.y_train[k_indices]
            # Conta manualmente as ocorrências de cada rótulo
            count_dict = {}
            for label in k_nearest_labels:
                if label in count_dict:
                    count_dict[label] += 1
                else:
                    count_dict[label] = 1
            # Determina o rótulo com a maior contagem (votação majoritária)
            max_label = None
            max_count = -1
            for label, count in count_dict.items():
                if count > max_count:
                    max_count = count
                    max_label = label
            predictions.append(max_label)
        return np.array(predictions)


# ==================== Funções de Geração de Dados ====================
def generate_two_class_data(n_points):
    """
    Gera dados sintéticos para um exemplo com duas classes.

    Parâmetros:
    -----------
    n_points : int
        Número de pontos a serem gerados para cada classe.

    Retorna:
    --------
    X : numpy array, shape = [2*n_points, 2]
        Dados gerados para as duas classes.
    y : numpy array, shape = [2*n_points]
        Rótulos dos dados (0 para a classe 0 e 1 para a classe 1).
    """
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)

    X0_list = X0.tolist()
    X1_list = X1.tolist()
    X = np.array(X0_list + X1_list)

    y0_list = y0.tolist()
    y1_list = y1.tolist()
    y = np.array(y0_list + y1_list)
    return X, y


def generate_three_class_data(n_points):
    """
    Gera dados sintéticos para um exemplo com três classes.

    Parâmetros:
    -----------
    n_points : int
        Número de pontos a serem gerados para cada classe.

    Retorna:
    --------
    X : numpy array, shape = [3*n_points, 2]
        Dados gerados para as três classes.
    y : numpy array, shape = [3*n_points]
        Rótulos dos dados (0, 1 e 2 para as três classes).
    """
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)
    X2 = np.random.randn(n_points, 2) + np.array([0, 5])
    y2 = np.full(n_points, 2)

    X0_list = X0.tolist()
    X1_list = X1.tolist()
    X2_list = X2.tolist()
    X = np.array(X0_list + X1_list + X2_list)

    y0_list = y0.tolist()
    y1_list = y1.tolist()
    y2_list = y2.tolist()
    y = np.array(y0_list + y1_list + y2_list)
    return X, y


# ==================== Funções de Plot ====================
def plot_data(X, y, title):
    """
    Plota os dados em um gráfico de dispersão.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)


def plot_decision_boundary(knn, X, y, title):
    """
    Plota a fronteira de decisão do modelo KNN junto com os dados.
    """
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = np.reshape(Z, np.shape(xx))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)


# ==================== Função Principal (Streamlit) ====================
def main():
    st.title("Aplicação Interativa do Modelo KNN")
    st.write("Ajuste os parâmetros na barra lateral e visualize os resultados.")

    # --------------------- Entradas via Sidebar ---------------------
    st.sidebar.header("Parâmetros")
    n_points = st.sidebar.number_input(
        "Quantidade de pontos para cada classe", min_value=10, value=50, step=10
    )
    k = st.sidebar.number_input("Número de vizinhos (k)", min_value=1, value=3, step=1)
    normalize = st.sidebar.radio("Normalizar dados?", options=["Sim", "Não"]) == "Sim"
    example_choice = st.sidebar.radio(
        "Escolha o exemplo", options=["Duas classes", "Três classes"]
    )

    # Opção para definir a fonte dos novos pontos: Manual ou Aleatória
    new_points_source = st.sidebar.radio(
        "Fonte dos novos pontos", options=["Manual", "Aleatória"]
    )

    if new_points_source == "Aleatória":
        n_new_points = st.sidebar.number_input(
            "Número de novos pontos aleatórios", min_value=1, value=4, step=1
        )
        # Um botão para gerar novos pontos aleatórios
        if st.sidebar.button("Gerar novos pontos aleatórios"):
            # Utiliza o range dos dados (X) para gerar pontos aleatórios com alguma margem
            # Se X ainda não foi gerado, usamos um range padrão
            x_range = (-1, 6)  # valores padrão caso X não exista ainda
            y_range = (-1, 6)
            new_points = np.column_stack(
                (
                    np.random.uniform(x_range[0], x_range[1], int(n_new_points)),
                    np.random.uniform(y_range[0], y_range[1], int(n_new_points)),
                )
            )
        else:
            # Se o botão não for clicado, deixamos new_points indefinido para não exibir nada ainda
            new_points = None
    else:
        # Fonte Manual: o usuário insere os pontos no formato "x1,y1; x2,y2; ..."
        new_points_str = st.sidebar.text_input(
            "Novos pontos (formato: x1,y1; x2,y2; ...)", value="1,1; 3,3; 0,4; 6,4"
        )
        try:
            new_points = np.array(
                [
                    [float(coord) for coord in point.split(",")]
                    for point in new_points_str.split(";")
                    if point.strip() != ""
                ]
            )
        except Exception as e:
            st.error("Erro ao processar os novos pontos. Verifique o formato.")
            new_points = np.array([[1, 1], [3, 3], [0, 4], [6, 4]])

    # --------------------- Geração dos Dados ---------------------
    if example_choice == "Três classes":
        X, y = generate_three_class_data(n_points)
        example_title = "Exemplo com Três Classes"
    else:
        X, y = generate_two_class_data(n_points)
        example_title = "Exemplo com Duas Classes"

    # --------------------- Exibição dos Gráficos ---------------------
    st.subheader(f"{example_title} - Dados Gerados (antes do modelo)")
    plot_data(X, y, f"{example_title} - Dados Gerados (antes do modelo)")

    # Treinamento do modelo KNN e plot da fronteira de decisão
    knn = KNNClassifier(k=k, normalize=normalize)
    knn.fit(X, y)
    st.subheader(
        f"{example_title} - Fronteira de Decisão (k = {k}, Normalização = {normalize})"
    )
    plot_decision_boundary(
        knn,
        X,
        y,
        f"{example_title} - Fronteira de Decisão (k = {k}, Normalização = {normalize})",
    )

    # --------------------- Exemplo Adicional de Previsão ---------------------
    st.subheader("Exemplo Adicional: Previsão para Novos Pontos")
    if new_points is None:
        st.info(
            "Clique no botão 'Gerar novos pontos aleatórios' para criar pontos aleatórios, ou insira manualmente os pontos."
        )
    else:
        predictions = knn.predict(new_points)
        # Cria uma tabela com as coordenadas e a previsão para cada novo ponto
        df = pd.DataFrame(
            {
                "Coordenadas": [f"({pt[0]:.2f}, {pt[1]:.2f})" for pt in new_points],
                "Previsão": predictions,
            }
        )
        st.write("Tabela de previsão dos novos pontos", df)

        # Plota os novos pontos sobre a fronteira de decisão
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = knn.predict(grid_points)
        Z = np.reshape(Z, np.shape(xx))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
        # Destaque para os novos pontos (marcados com "x" e cor vermelha)
        ax.scatter(
            new_points[:, 0],
            new_points[:, 1],
            c="red",
            s=100,
            marker="x",
            label="Novos Pontos",
        )
        ax.set_title("Exemplo Adicional: Previsão para Novos Pontos")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        st.pyplot(fig)


# Executa a função principal quando o script é executado via Streamlit
if __name__ == "__main__":
    main()

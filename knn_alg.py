import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Classe KNNClassifier
# =============================================================================
class KNNClassifier:
    """
    Classe que implementa o algoritmo KNN para classificação.
    """

    def __init__(self, k=3, normalize=True):
        """
        Inicializa o classificador.

        Parâmetros:
        -----------
        k : int
            Número de vizinhos a serem considerados na classificação.
        normalize : bool
            Se True, normaliza os dados com min-max scaling.
        """
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.min = None  # Valores mínimos para normalização
        self.max = None  # Valores máximos para normalização

    def fit(self, X, y):
        """
        Armazena os dados de treinamento e normaliza, se necessário.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.normalize:
            self.min = self.X_train.min(axis=0)
            self.max = self.X_train.max(axis=0)
            diff = self.max - self.min
            diff[diff == 0] = 1  # Evita divisão por zero
            self.X_train = (self.X_train - self.min) / diff

    def predict(self, X):
        """
        Prediz os rótulos para os dados de teste.
        """
        X = np.array(X)
        if self.normalize:
            diff = self.max - self.min
            diff[diff == 0] = 1
            X = (X - self.min) / diff

        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = distances.argsort()[: self.k]
            k_nearest_labels = self.y_train[k_indices]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = labels[np.argmax(counts)]
            predictions.append(prediction)
        return np.array(predictions)


# =============================================================================
# Funções para gerar dados sintéticos
# =============================================================================
def generate_two_class_data(n_points):
    """
    Gera dados para um exemplo com duas classes.
    """
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    return X, y


def generate_three_class_data(n_points):
    """
    Gera dados para um exemplo com três classes.
    """
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)
    X2 = np.random.randn(n_points, 2) + np.array([0, 5])
    y2 = np.full(n_points, 2)
    X = np.vstack((X0, X1, X2))
    y = np.hstack((y0, y1, y2))
    return X, y


# =============================================================================
# Funções de plotagem
# =============================================================================
def plot_data(X, y, title):
    """
    Cria um gráfico de dispersão dos dados.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig


def plot_decision_boundary(knn, X, y, title):
    """
    Cria o gráfico da fronteira de decisão do modelo KNN.
    """
    h = 0.1  # tamanho do passo para a grade
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig


# =============================================================================
# Função principal (interface Streamlit)
# =============================================================================
def main():
    st.title("KNN - Classificador com Streamlit")
    st.write("Ajuste os parâmetros utilizando os controles na barra lateral.")

    # --------------------- Entrada de Parâmetros via Sidebar ---------------------
    n_points = st.sidebar.number_input(
        "Quantidade de pontos por classe:",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
    )

    k = st.sidebar.number_input(
        "Número de vizinhos (k):", min_value=1, max_value=20, value=3, step=1
    )

    normalize = st.sidebar.checkbox("Normalizar dados?", value=True)

    example_choice = st.sidebar.radio(
        "Escolha o exemplo:", options=["Duas Classes", "Três Classes"]
    )

    # --------------------- Geração dos Dados ---------------------
    if example_choice == "Três Classes":
        X, y = generate_three_class_data(n_points)
        example_title = "Exemplo com Três Classes"
    else:
        X, y = generate_two_class_data(n_points)
        example_title = "Exemplo com Duas Classes"

    # Exibe o gráfico dos dados gerados
    fig_data = plot_data(X, y, f"{example_title} - Dados Gerados (antes do modelo)")
    st.pyplot(fig_data)

    # --------------------- Treinamento e Fronteira de Decisão ---------------------
    knn = KNNClassifier(k=k, normalize=normalize)
    knn.fit(X, y)
    fig_boundary = plot_decision_boundary(
        knn,
        X,
        y,
        f"{example_title} - Fronteira de Decisão (k = {k}, Normalização = {normalize})",
    )
    st.pyplot(fig_boundary)

    # --------------------- Exemplo Adicional de Previsão ---------------------
    new_points = np.array([[1, 1], [3, 3], [0, 4], [5, 0]])
    predictions = knn.predict(new_points)
    st.write("Previsões para novos pontos:", predictions)

    # Plota os novos pontos sobre a fronteira de decisão para visualização
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
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


if __name__ == "__main__":
    main()

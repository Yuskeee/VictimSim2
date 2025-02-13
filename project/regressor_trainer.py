import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

def load_dataset(filepath):
    """
    Carrega o dataset a partir de um arquivo de texto sem cabeçalho.
    Ajuste o separador se necessário.
    """
    df = pd.read_csv(filepath, sep=',', header=None)
    return df

def main():
    # ---------------------------
    # Treinamento/Validação (4000 vítimas)
    # ---------------------------
    dataset_path = "datasets/data_4000v/env_vital_signals.txt"
    data = load_dataset(dataset_path)
    
    # Seleciona as features permitidas: qPA (coluna 3), pulso (coluna 4) e freq_resp (coluna 5)
    features = data.iloc[:, [3, 4, 5]].values
    target = data.iloc[:, 6].values

    # Divisão: 80% treino, 20% validação
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, shuffle=True, random_state=42)

    print(f"Tam. total do dataset.......: {len(features):>4}")
    print(f"Tam. dataset treino & valid.: {len(X_train):>4}")
    print(f"Tam. dataset testes cegos...: {len(X_val):>4}\n")

    # Escalonamento das features para MLP e DecisionTree
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    k_fold = 5         # Número de folds para validação cruzada
    n_param = 3        # Número de parametrizações

    # DecisionTreeRegressor----------------------------------------------
    # Parametrizações:
    # Underfitting: árvore muito rasa (max_depth=2, min_samples_split=10, min_samples_leaf=4)
    # Balanced: complexidade moderada (max_depth=40, min_samples_split=3, min_samples_leaf=2)
    # Overfitting: árvore muito profunda (max_depth=100, min_samples_split=2, min_samples_leaf=1)
    max_depths         = [2, 40, 100]
    min_samples_splits = [10, 3, 2]
    min_samples_leafs  = [4, 2, 1]
    
    dt_models = []      
    dt_mse = []         

    for i in range(n_param):
        dt = DecisionTreeRegressor(random_state=42,
                                   max_depth=max_depths[i],
                                   min_samples_split=min_samples_splits[i],
                                   min_samples_leaf=min_samples_leafs[i])
        scores = cross_validate(dt, X_train_scaled, y_train, cv=k_fold,
                                scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'),
                                return_train_score=True,
                                return_estimator=True)
        rmse_val = np.sqrt(-scores['test_neg_mean_squared_error']).mean()

        best_fold_idx = np.argmin(np.abs(scores['train_neg_mean_squared_error'] - scores['test_neg_mean_squared_error']))
        fitted_dt = scores['estimator'][best_fold_idx]
        dt_models.append(fitted_dt)
        dt_mse.append(rmse_val)
        
        print(f"DecisionTreeRegressor - Parametrização {i+1}:")
        print(f"  max_depth={max_depths[i]}, min_samples_split={min_samples_splits[i]}, min_samples_leaf={min_samples_leafs[i]}")
        print(f"  RMSE (treino): {np.sqrt(-scores['train_neg_mean_squared_error']).mean():.2f}")
        print(f"  RMSE (validação): {rmse_val:.2f}\n")
    
    best_index_dt = dt_mse.index(min(dt_mse))
    best_dt_model = dt_models[best_index_dt]
    print(f"Melhor parametrização DT: {best_index_dt+1} com RMSE de validação: {dt_mse[best_index_dt]:.2f}\n")
    
    # MLPRegressor------------------------------------------------------
    
    # Parametrizações:
    # Underfitting: rede pequena, (10,), learning_rate_init=0.01, momentum=0.0, activation='relu', solver='sgd', max_iter=1000
    # Balanced: rede moderada, (50,), learning_rate_init=0.05, momentum=0.9, activation='relu', solver='adam', max_iter=5000
    # Overfitting: rede grande, (100, 100), learning_rate_init=0.1, momentum=0.9, activation='tanh', solver='adam', max_iter=10000
    hidden_layers   = [(10,), (50,), (100, 100)]
    learning_rates  = [0.01, 0.05, 0.1]
    momentums       = [0.0, 0.9, 0.9]
    activations     = ['relu', 'relu', 'tanh']
    solvers         = ['sgd', 'adam', 'adam']
    max_iters       = [1000, 5000, 10000]
    
    mlp_models = []
    mlp_mse = []
    
    for i in range(n_param):
        mlp = MLPRegressor(random_state=42,
                           hidden_layer_sizes=hidden_layers[i],
                           learning_rate_init=learning_rates[i],
                           momentum=momentums[i],
                           activation=activations[i],
                           solver=solvers[i],
                           max_iter=max_iters[i])

        scores = cross_validate(mlp, X_train_scaled, y_train, cv=k_fold,
                                scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'),
                                return_train_score=True,
                                return_estimator=True)
        rmse_val = np.sqrt(-scores['test_neg_mean_squared_error']).mean()
        best_fold_idx = np.argmin(np.abs(scores['train_neg_mean_squared_error'] - scores['test_neg_mean_squared_error']))
        fitted_mlp = scores['estimator'][best_fold_idx]
        mlp_models.append(fitted_mlp)
        mlp_mse.append(rmse_val)
        
        print(f"MLPRegressor - Parametrização {i+1}:")
        print(f"  hidden_layer_sizes={hidden_layers[i]}, learning_rate_init={learning_rates[i]}, momentum={momentums[i]}")
        print(f"  activation={activations[i]}, solver={solvers[i]}, max_iter={max_iters[i]}")
        print(f"  RMSE (treino): {np.sqrt(-scores['train_neg_mean_squared_error']).mean():.2f}")
        print(f"  RMSE (validação): {rmse_val:.2f}\n")
    
    best_index_mlp = mlp_mse.index(min(mlp_mse))
    best_mlp_model = mlp_models[best_index_mlp]
    print(f"Melhor parametrização MLP: {best_index_mlp+1} com RMSE de validação: {mlp_mse[best_index_mlp]:.2f}\n")
    
    # Seleção do melhor modelo global (com base no RMSE da validação cruzada)
    if dt_mse[best_index_dt] < mlp_mse[best_index_mlp]:
        best_model = best_dt_model
        best_model_name = "DecisionTreeRegressor"
        best_val_rmse = dt_mse[best_index_dt]
    else:
        best_model = best_mlp_model
        best_model_name = "MLPRegressor"
        best_val_rmse = mlp_mse[best_index_mlp]
    print(f"Modelo selecionado para integração: {best_model_name} com RMSE de validação: {best_val_rmse:.2f}\n")
    
    # Salva o melhor modelo
    with open("regressor.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Melhor modelo salvo em 'regressor.pkl'.\n")
    
    ##########################################################################

    # Pré-teste Cego: Teste com os dois modelos no dataset de 800 vítimas
    blind_dataset_path = "datasets/data_800v/env_vital_signals.txt"
    test_data = load_dataset(blind_dataset_path)
    
    # Seleciona as mesmas features e variável alvo
    X_test = test_data.iloc[:, [3, 4, 5]].values
    y_test = test_data.iloc[:, 6].values
    
    # Para MLP, escala o conjunto de teste com o scaler treinado
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Tamanho do dataset de testes cegos (800 vítimas): {len(X_test)}\n")
    
    # Aplica o modelo DecisionTreeRegressor no teste cego (não necessita de escalonamento)
    dt_blind_pred = best_dt_model.predict(X_test_scaled)
    dt_blind_rmse = np.sqrt(mean_squared_error(y_test, dt_blind_pred))
    dt_blind_mae = mean_absolute_error(y_test, dt_blind_pred)
    dt_blind_r2 = r2_score(y_test, dt_blind_pred)
    
    print("Pré-teste Cego com DecisionTreeRegressor:")
    print(f"  RMSE: {dt_blind_rmse:.2f}")
    print(f"  MAE:  {dt_blind_mae:.2f}")
    print(f"  R2:   {dt_blind_r2:.2f}\n")
    
    # Aplica o modelo MLPRegressor no teste cego
    mlp_blind_pred = best_mlp_model.predict(X_test_scaled)
    mlp_blind_rmse = np.sqrt(mean_squared_error(y_test, mlp_blind_pred))
    mlp_blind_mae = mean_absolute_error(y_test, mlp_blind_pred)
    mlp_blind_r2 = r2_score(y_test, mlp_blind_pred)
    
    print("Pré-teste Cego com MLPRegressor:")
    print(f"  RMSE: {mlp_blind_rmse:.2f}")
    print(f"  MAE:  {mlp_blind_mae:.2f}")
    print(f"  R2:   {mlp_blind_r2:.2f}")

if __name__ == "__main__":
    main()

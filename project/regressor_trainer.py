import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def load_dataset(filepath):
    """
    Carrega o dataset a partir de um arquivo de texto sem cabeçalho.
    Ajuste o separador se necessário.
    """
    df = pd.read_csv(filepath, sep=',', header=None)
    return df

def main():
    # Carrega o dataset de 4000 vítimas (treinamento/validação)
    dataset_path = "datasets/data_4000v/env_vital_signals.txt"
    data = load_dataset(dataset_path)
    
    # Seleciona as features permitidas para predição:
    # qPA (coluna 3), pulso (coluna 4) e freq_resp (coluna 5)
    features = data.iloc[:, [3, 4, 5]]
    
    # Seleciona a variável alvo: gravidade (coluna 6)
    target = data.iloc[:, 6]
    
    # Cria as partições para o grid search: 70% para treinamento e 30% para validação
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.3, shuffle=True, random_state=42)

    print(f"Tam. total do dataset.......:\t{len(features):>4}")
    print(f"Tam. dataset treino & valid.:\t{len(X_train):>4}")
    print(f"Tam. dataset testes cegos...:\t{len(X_val):>4}")

    k_fold = 5
    
    # Ajuste do DecisionTreeRegressor (CART) via Grid Search
    dt_param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = DecisionTreeRegressor(random_state=42)
    dt_grid = GridSearchCV(dt, dt_param_grid, scoring='neg_mean_squared_error', cv=k_fold, n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    dt_best = dt_grid.best_estimator_
    print("DecisionTreeRegressor - Melhores parâmetros:", dt_grid.best_params_)
    
    # Avaliação no conjunto de validação
    dt_pred_val = dt_best.predict(X_val)
    dt_rmse = np.sqrt(mean_squared_error(y_val, dt_pred_val))
    dt_mae = mean_absolute_error(y_val, dt_pred_val)
    dt_r2 = r2_score(y_val, dt_pred_val)
    print("DecisionTreeRegressor - Validação:")
    print(f"  RMSE: {dt_rmse:.2f}")
    print(f"  MAE: {dt_mae:.2f}")
    print(f"  R2: {dt_r2:.2f}")
    
    # Validação cruzada com todo o dataset para o DecisionTreeRegressor
    dt_cv_results = cross_validate(dt_best, features, target, cv=k_fold,
                                   scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'),
                                   return_train_score=False)
    dt_cv_rmse = np.sqrt(-np.mean(dt_cv_results['test_neg_mean_squared_error']))
    dt_cv_mae = -np.mean(dt_cv_results['test_neg_mean_absolute_error'])
    dt_cv_r2 = np.mean(dt_cv_results['test_r2'])
    print(f"DecisionTreeRegressor - Validação Cruzada ({k_fold}-fold):")
    print(f"  CV RMSE: {dt_cv_rmse:.2f}")
    print(f"  CV MAE: {dt_cv_mae:.2f}")
    print(f"  CV R2: {dt_cv_r2:.2f}")
    
    # Ajuste do MLPRegressor (Rede Neural) via Grid Search
    mlp_param_grid = {
        'hidden_layer_sizes': [(10,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.005, 0.01],
        'learning_rate_init': [0.01, 0.05, 0.1]
    }
    mlp = MLPRegressor(max_iter=5000, random_state=42)
    mlp_grid = GridSearchCV(mlp, mlp_param_grid, scoring='neg_mean_squared_error', cv=k_fold, n_jobs=-1)
    mlp_grid.fit(X_train, y_train)
    mlp_best = mlp_grid.best_estimator_
    print("MLPRegressor - Melhores parâmetros:", mlp_grid.best_params_)
    
    # Avaliação no conjunto de validação
    mlp_pred_val = mlp_best.predict(X_val)
    mlp_rmse = np.sqrt(mean_squared_error(y_val, mlp_pred_val))
    mlp_mae = mean_absolute_error(y_val, mlp_pred_val)
    mlp_r2 = r2_score(y_val, mlp_pred_val)
    print("MLPRegressor - Validação:")
    print(f"  RMSE: {mlp_rmse:.2f}")
    print(f"  MAE: {mlp_mae:.2f}")
    print(f"  R2: {mlp_r2:.2f}")
    
    # Validação cruzada com todo o dataset para o MLPRegressor
    mlp_cv_results = cross_validate(mlp_best, features, target, cv=k_fold,
                                    scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'),
                                    return_train_score=False)
    mlp_cv_rmse = np.sqrt(-np.mean(mlp_cv_results['test_neg_mean_squared_error']))
    mlp_cv_mae = -np.mean(mlp_cv_results['test_neg_mean_absolute_error'])
    mlp_cv_r2 = np.mean(mlp_cv_results['test_r2'])
    print(f"MLPRegressor - Validação Cruzada ({k_fold}-fold):")
    print(f"  CV RMSE: {mlp_cv_rmse:.2f}")
    print(f"  CV MAE: {mlp_cv_mae:.2f}")
    print(f"  CV R2: {mlp_cv_r2:.2f}")
    
    # Seleção do melhor modelo com base no RMSE no conjunto de validação (talvez seja melhor pelo R2? Ou avaliar com múltiplas métricas?)
    if dt_cv_rmse < mlp_cv_rmse:
        best_model = dt_best
        best_model_name = "DecisionTreeRegressor"
        best_val_rmse = dt_cv_rmse
    else:
        best_model = mlp_best
        best_model_name = "MLPRegressor"
        best_val_rmse = mlp_cv_rmse
    print(f"Modelo selecionado: {best_model_name} com RMSE de validação: {best_val_rmse:.2f}")
    
    # Salva o melhor modelo em um arquivo pickle
    with open("regressor.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Melhor modelo salvo em 'regressor.pkl'.")

if __name__ == "__main__":
    main()

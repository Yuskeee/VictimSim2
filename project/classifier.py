import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

##############################################################################
#                   FUNÇÕES GERAIS DE MÉTRICAS E AVALIAÇÃO                   #
##############################################################################

def calc_metrics(y_true, y_pred):
    """
    Para classificação, calculamos a acurácia e F1 (macro ou weighted).
    Se quiser outras métricas, basta adicionar aqui.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

##############################################################################
#          CONFIGURAÇÕES: ÁRVORE vs. REDE NEURAL (apenas CLASSIFICADOR)      #
##############################################################################

# 3 configurações para Decision Tree Classifier
CART_CONFIGS = [
    {'max_depth': 3, 'min_samples_leaf': 5},
    {'max_depth': 10, 'min_samples_leaf': 2},
    {'max_depth': None, 'min_samples_leaf': 1},
]

# 3 configurações para Rede Neural (classificador)
MLP_CONFIGS = [
    {'hidden_layers': [32, 16],       'lr': 0.001, 'epochs': 50},
    {'hidden_layers': [64, 32],       'lr': 0.001, 'epochs': 100},
    {'hidden_layers': [64, 64, 32],   'lr': 0.0005, 'epochs': 150},
]

##############################################################################
#                       FUNÇÕES DE CRIAÇÃO DE MODELOS                        #
##############################################################################

def create_cart_classifier(config):
    """
    Cria um DecisionTreeClassifier dado o dicionário de hiperparâmetros.
    """
    model = DecisionTreeClassifier(
        max_depth=config['max_depth'],
        min_samples_leaf=config['min_samples_leaf'],
        random_state=42
    )
    return model

def create_mlp_classifier(input_dim, config):
    """
    Cria e compila uma rede neural para classificação de 4 classes.
    """
    model = Sequential()
    # Camadas ocultas
    for i, units in enumerate(config['hidden_layers']):
        if i == 0:
            model.add(Dense(units, input_dim=input_dim, activation='relu'))
        else:
            model.add(Dense(units, activation='relu'))

    # Camada de saída: 4 neurônios (softmax) para 4 classes
    model.add(Dense(4, activation='softmax'))

    opt = Adam(learning_rate=config['lr'])
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',  # pois y são classes inteiras [1..4]
        metrics=['accuracy']
    )
    return model

##############################################################################
#               FUNÇÃO GERAL DE CROSS-VALIDATION PARA CLASSIF                #
##############################################################################

def cross_validate_model(X, y, model_type, configs, n_splits=5):
    """
    Executa cross-validation (KFold) para cada configuração do model_type ('CART' ou 'MLP').
    Retorna uma lista com dicionários de resultados, um por config.
    """
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for config_id, config in enumerate(configs):
        all_acc, all_f1 = [], []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            if model_type == 'CART':
                model = create_cart_classifier(config)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

            elif model_type == 'MLP':
                input_dim = X.shape[1]
                model = create_mlp_classifier(input_dim, config)
                # Keras espera y_train no formato (n,) com valores 0..3
                # Se suas classes forem 1..4, talvez seja preciso subtrair 1 para ficar 0..3:
                # y_train_adj = y_train - 1
                # y_val_adj = y_val - 1
                # Ajuste caso necessário!
                # Aqui assumindo que suas classes já são 0..3 ou usando adequações equivalentes
                model.fit(X_train, y_train,
                          epochs=config['epochs'],
                          batch_size=32,
                          verbose=0)
                y_pred = model.predict(X_val)
                # y_pred é probabilidade [N,4], pegue argmax
                y_pred = np.argmax(y_pred, axis=1)

            # Calcula métricas (acc, f1)
            acc, f1 = calc_metrics(y_val, y_pred)
            all_acc.append(acc)
            all_f1.append(f1)

        mean_acc = np.mean(all_acc)
        mean_f1  = np.mean(all_f1)

        results.append({
            'config_id': config_id,
            'config': config,
            'mean_acc': mean_acc,
            'mean_f1':  mean_f1
        })

    return results

def find_best_config(results, metric='mean_acc'):
    """
    Encontra a configuração com melhor valor de 'metric' (por default, accuracy).
    """
    best = None
    best_val = -9999
    for res in results:
        val = res[metric]
        if val > best_val:
            best_val = val
            best = res
    return best


##############################################################################
#                     TREINANDO O MODELO FINAL ESCOLHIDO                     #
##############################################################################

def train_final_model(X, y, model_type, best_config):
    """
    Treina um modelo final (com TODOS os dados) usando a melhor configuração.
    """
    if model_type == 'CART':
        model = create_cart_classifier(best_config['config'])
        model.fit(X, y)
    elif model_type == 'MLP':
        input_dim = X.shape[1]
        model = create_mlp_classifier(input_dim, best_config['config'])
        model.fit(X, y, epochs=best_config['config']['epochs'], batch_size=32, verbose=0)
    return model

##############################################################################
#                   SALVAR / CARREGAR (PARA USAR NO RESCUER)                 #
##############################################################################

def load_model(filename='best_model.pkl'):
    """
    Carrega o modelo final escolhido (classificador).
    """
    return joblib.load(filename)

##############################################################################
#                                TESTE CEGO                                  #
##############################################################################
def test_model(dataset_path):
    # Leitura do dataset de 800 vítimas (novamente, sem header)
    df_test = pd.read_csv(dataset_path, sep=",", header=None)
    X_test = df_test.iloc[:, [3, 4, 5]]
    y_test = df_test.iloc[:, 7] - 1

    # Previsão usando o modelo final
    if final_winner_type == 'MLP':
        y_pred_test = best_model.predict(X_test)
        y_pred_test = np.argmax(y_pred_test, axis=1)
    else:
        y_pred_test = best_model.predict(X_test)

    # Cálculo das métricas no conjunto de teste
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    recall_test = recall_score(y_test, y_pred_test, average='macro')
    precision_test = precision_score(y_test, y_pred_test, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    print("\n=== Teste Cego no dataset ===")
    print(f"Test Accuracy: {acc_test:.3f}")
    print(f"Test F1 Score: {f1_test:.3f}")
    print(f"Test Recall: {recall_test:.3f}")
    print(f"Test Precision: {precision_test:.3f}")
    print("Confusion Matrix:")
    print(conf_matrix)

##############################################################################
#                               MAIN                                  #
##############################################################################

if __name__ == "__main__":
    # Exemplo de leitura dos dados
    df = pd.read_csv('datasets/data_4000v/env_vital_signals.txt', sep=",", header=None)

    # Seleciona apenas as features que serão utilizadas no classificador
    X = df.iloc[:, [3, 4, 5]]
    y = df.iloc[:, 7] - 1  # Ajusta os rótulos de 1..4 para 0..3


    print("=== CROSS-VALIDATION - DecisionTreeClassifier ===")
    cart_results = cross_validate_model(X, y, 'CART', CART_CONFIGS, n_splits=5)
    for r in cart_results:
        print(f"Config CART {r['config_id']} -> ACC={r['mean_acc']:.3f} | F1={r['mean_f1']:.3f}")

    best_cart = find_best_config(cart_results, metric='mean_acc')
    print("Melhor config CART:", best_cart)

    print("\n=== CROSS-VALIDATION - MLP Classifier ===")
    mlp_results = cross_validate_model(X, y, 'MLP', MLP_CONFIGS, n_splits=5)
    for r in mlp_results:
        print(f"Config MLP {r['config_id']} -> ACC={r['mean_acc']:.3f} | F1={r['mean_f1']:.3f}")

    best_mlp = find_best_config(mlp_results, metric='mean_acc')
    print("Melhor config MLP:", best_mlp)

    # Compara as duas melhores (CART vs MLP)
    if best_cart['mean_acc'] > best_mlp['mean_acc']:
        final_winner_type = 'CART'
        final_winner = best_cart
    else:
        final_winner_type = 'MLP'
        final_winner = best_mlp

    print(f"\nMelhor algoritmo final: {final_winner_type}")
    print("Dados da melhor config:", final_winner)

    # Treina o modelo final com TODOS os dados
    best_model = train_final_model(X, y, final_winner_type, final_winner)

    # Salva para usar no Rescuer
    joblib.dump(best_model, 'best_model.pkl')
    print("\nModelo final salvo em 'best_model.pkl'")

    test_model('datasets/data_800v/env_vital_signals.txt');

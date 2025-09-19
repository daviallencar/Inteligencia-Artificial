import numpy as np
import matplotlib.pyplot as plt

def carregar_dados():
    """Carrega os dados do EMG"""
    print("Carregando dados do EMG...")
    data = np.loadtxt("data/EMGsDataset.csv", delimiter=',')
    data = data.T  # Transpor para ter as características nas colunas
    print(f"Dados carregados: {data.shape[0]} observações, {data.shape[1]-1} características")
    return data

def organizar_dados(data):
    """1. Organizar dados em variáveis X e Y"""
    print("\n=== 1. ORGANIZAÇÃO DOS DADOS ===")
    
    N, p_plus_1 = data.shape
    p = p_plus_1 - 1  # Número de características
    
    # Para MQO: X (N×p), Y (N×C)
    X_mqo = data[:, :-1]  # Todas as colunas exceto a última
    y_labels = data[:, -1].astype(int)  # Última coluna (classes)
    
    # Criar matriz Y one-hot para MQO
    classes = np.unique(y_labels)
    C = len(classes)
    Y_mqo = np.zeros((N, C))
    for i, classe in enumerate(classes):
        Y_mqo[y_labels == classe, i] = 1
    
    # Para modelos gaussianos: X (p×N), Y (C×N)
    X_gauss = X_mqo.T
    Y_gauss = Y_mqo.T
    
    print(f"Para MQO:")
    print(f"  X: {X_mqo.shape} (N×p)")
    print(f"  Y: {Y_mqo.shape} (N×C)")
    print(f"Para Modelos Gaussianos:")
    print(f"  X: {X_gauss.shape} (p×N)")
    print(f"  Y: {Y_gauss.shape} (C×N)")
    print(f"Classes encontradas: {classes}")
    print(f"Número de classes: {C}")
    
    return X_mqo, Y_mqo, X_gauss, Y_gauss, y_labels, classes

def visualizacao_inicial(X, y_labels, classes):
    """2. Visualização inicial dos dados"""
    print("\n=== 2. VISUALIZAÇÃO INICIAL DOS DADOS ===")
    
    # Nomes das classes
    nomes_classes = ["Neutro", "Sorriso", "Sobrancelhas levantadas", "Surpreso", "Rabugento"]
    cores = ["blue", "red", "green", "orange", "purple"]
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico de espalhamento
    for i, classe in enumerate(classes):
        mask = y_labels == classe
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=cores[i], alpha=0.6, s=20, 
                   label=nomes_classes[i], edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Sensor 1 (Corrugador do Supercílio)')
    plt.ylabel('Sensor 2 (Zigomático Maior)')
    plt.title('Dados EMG - Classificação de Expressões Faciais')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Estatísticas por classe
    print("\nEstatísticas por classe:")
    for i, classe in enumerate(classes):
        mask = y_labels == classe
        count = np.sum(mask)
        print(f"  {nomes_classes[i]}: {count} amostras ({count/len(y_labels)*100:.1f}%)")
    
    # Discussão sobre separabilidade
    print("\nDISCUSSÃO SOBRE SEPARABILIDADE:")
    print("• Os dados parecem ter alguma separabilidade linear")
    print("• Algumas classes podem ser separadas por retas")
    print("• Modelos lineares (MQO) podem ter bom desempenho")
    print("• Modelos gaussianos podem capturar melhor a distribuição dos dados")
    print("• Classes podem ter sobreposição, exigindo modelos mais sofisticados")

def mqo_tradicional(X_train, Y_train):
    """MQO tradicional para classificação"""
    # Adicionar intercepto
    X_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train])
    # Calcular pesos
    W = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ Y_train
    return W

def predicao_mqo(X_test, W):
    """Predição para MQO"""
    X_test_intercept = np.column_stack([np.ones(X_test.shape[0]), X_test])
    scores = X_test_intercept @ W
    return np.argmax(scores, axis=1)

def gaussiano_tradicional(X_train, Y_train):
    """Classificador Gaussiano Tradicional"""
    N, p = X_train.shape
    C = Y_train.shape[1]
    
    # Calcular parâmetros para cada classe
    parametros = {}
    for c in range(C):
        mask = Y_train[:, c] == 1
        X_c = X_train[mask]
        
        if len(X_c) > 0:
            mu_c = np.mean(X_c, axis=0)
            Sigma_c = np.cov(X_c.T)
            # Adicionar pequeno valor para evitar singularidade
            Sigma_c += np.eye(Sigma_c.shape[0]) * 1e-6
            pi_c = len(X_c) / N
            
            parametros[c] = {
                'mu': mu_c,
                'Sigma': Sigma_c,
                'pi': pi_c
            }
    
    return parametros

def gaussiano_covariancias_iguais(X_train, Y_train):
    """Classificador Gaussiano com Covariâncias Iguais"""
    N, p = X_train.shape
    C = Y_train.shape[1]
    
    # Calcular parâmetros para cada classe
    parametros = {}
    Sigma_pooled = np.zeros((p, p))
    
    for c in range(C):
        mask = Y_train[:, c] == 1
        X_c = X_train[mask]
        
        if len(X_c) > 0:
            mu_c = np.mean(X_c, axis=0)
            pi_c = len(X_c) / N
            
            # Contribuição para covariância pooled
            Sigma_c = np.cov(X_c.T)
            Sigma_pooled += (len(X_c) - 1) * Sigma_c
            
            parametros[c] = {
                'mu': mu_c,
                'pi': pi_c
            }
    
    # Covariância pooled
    Sigma_pooled = Sigma_pooled / (N - C)
    # Adicionar pequeno valor para evitar singularidade
    Sigma_pooled += np.eye(Sigma_pooled.shape[0]) * 1e-6
    
    # Adicionar covariância pooled a todos os parâmetros
    for c in parametros:
        parametros[c]['Sigma'] = Sigma_pooled
    
    return parametros

def gaussiano_matriz_agregada(X_train, Y_train):
    """Classificador Gaussiano com Matriz Agregada"""
    N, p = X_train.shape
    C = Y_train.shape[1]
    
    # Calcular parâmetros para cada classe
    parametros = {}
    Sigma_agregada = np.zeros((p, p))
    
    for c in range(C):
        mask = Y_train[:, c] == 1
        X_c = X_train[mask]
        
        if len(X_c) > 0:
            mu_c = np.mean(X_c, axis=0)
            pi_c = len(X_c) / N
            
            # Contribuição para matriz agregada
            Sigma_c = np.cov(X_c.T)
            Sigma_agregada += pi_c * Sigma_c
            
            parametros[c] = {
                'mu': mu_c,
                'pi': pi_c
            }
    
    # Adicionar pequeno valor para evitar singularidade
    Sigma_agregada += np.eye(Sigma_agregada.shape[0]) * 1e-6
    
    # Adicionar matriz agregada a todos os parâmetros
    for c in parametros:
        parametros[c]['Sigma'] = Sigma_agregada
    
    return parametros

def gaussiano_regularizado(X_train, Y_train, lambda_reg):
    """Classificador Gaussiano Regularizado (Friedman)"""
    N, p = X_train.shape
    C = Y_train.shape[1]
    
    # Calcular parâmetros para cada classe
    parametros = {}
    Sigma_pooled = np.zeros((p, p))
    
    for c in range(C):
        mask = Y_train[:, c] == 1
        X_c = X_train[mask]
        
        if len(X_c) > 0:
            mu_c = np.mean(X_c, axis=0)
            pi_c = len(X_c) / N
            
            # Contribuição para covariância pooled
            Sigma_c = np.cov(X_c.T)
            Sigma_pooled += (len(X_c) - 1) * Sigma_c
            
            parametros[c] = {
                'mu': mu_c,
                'pi': pi_c
            }
    
    # Covariância pooled
    Sigma_pooled = Sigma_pooled / (N - C)
    
    # Regularização de Friedman
    for c in parametros:
        Sigma_c = parametros[c].get('Sigma', np.cov(X_train[Y_train[:, c] == 1].T) if np.sum(Y_train[:, c]) > 1 else np.eye(p))
        Sigma_reg = (1 - lambda_reg) * Sigma_c + lambda_reg * Sigma_pooled
        # Adicionar pequeno valor para evitar singularidade
        Sigma_reg += np.eye(Sigma_reg.shape[0]) * 1e-6
        parametros[c]['Sigma'] = Sigma_reg
    
    return parametros

def bayes_ingenuo(X_train, Y_train):
    """Classificador de Bayes Ingênuo"""
    N, p = X_train.shape
    C = Y_train.shape[1]
    
    # Calcular parâmetros para cada classe
    parametros = {}
    
    for c in range(C):
        mask = Y_train[:, c] == 1
        X_c = X_train[mask]
        
        if len(X_c) > 0:
            mu_c = np.mean(X_c, axis=0)
            var_c = np.var(X_c, axis=0)  # Variâncias individuais (diagonal)
            # Adicionar pequeno valor para evitar variância zero
            var_c = np.maximum(var_c, 1e-10)
            pi_c = len(X_c) / N
            
            parametros[c] = {
                'mu': mu_c,
                'var': var_c,  # Apenas variâncias (covariância diagonal)
                'pi': pi_c
            }
    
    return parametros

def predicao_gaussiana(X_test, parametros):
    """Predição para modelos gaussianos"""
    N_test = X_test.shape[0]
    C = len(parametros)
    log_probs = np.zeros((N_test, C))
    
    for c in range(C):
        if c in parametros:
            mu = parametros[c]['mu']
            Sigma = parametros[c]['Sigma']
            pi = parametros[c]['pi']
            
            # Calcular log da probabilidade
            diff = X_test - mu
            try:
                # Adicionar pequeno valor para evitar singularidade
                Sigma_safe = Sigma + np.eye(Sigma.shape[0]) * 1e-6
                inv_Sigma = np.linalg.inv(Sigma_safe)
                log_det = np.log(np.linalg.det(Sigma_safe))
                
                # Log da densidade gaussiana
                log_prob = -0.5 * np.sum(diff @ inv_Sigma * diff, axis=1)
                log_prob -= 0.5 * log_det
                log_prob -= 0.5 * len(mu) * np.log(2 * np.pi)
                log_prob += np.log(pi)
                
                log_probs[:, c] = log_prob
            except np.linalg.LinAlgError:
                # Se ainda houver problema, usar pseudo-inversa
                inv_Sigma = np.linalg.pinv(Sigma)
                log_det = np.log(np.linalg.det(Sigma + 1e-6 * np.eye(len(mu))))
                
                log_prob = -0.5 * np.sum(diff @ inv_Sigma * diff, axis=1)
                log_prob -= 0.5 * log_det
                log_prob -= 0.5 * len(mu) * np.log(2 * np.pi)
                log_prob += np.log(pi)
                
                log_probs[:, c] = log_prob
    
    return np.argmax(log_probs, axis=1)

def predicao_bayes_ingenuo(X_test, parametros):
    """Predição para Bayes Ingênuo"""
    N_test = X_test.shape[0]
    C = len(parametros)
    log_probs = np.zeros((N_test, C))
    
    for c in range(C):
        if c in parametros:
            mu = parametros[c]['mu']
            var = parametros[c]['var']
            pi = parametros[c]['pi']
            
            # Log da probabilidade (assumindo independência)
            diff = X_test - mu
            # Adicionar pequeno valor para evitar divisão por zero
            var_safe = np.maximum(var, 1e-10)
            log_prob = -0.5 * np.sum((diff**2) / var_safe, axis=1)
            log_prob -= 0.5 * np.sum(np.log(2 * np.pi * var_safe))
            log_prob += np.log(pi)
            
            log_probs[:, c] = log_prob
    
    return np.argmax(log_probs, axis=1)

def train_test_split_numpy(X, y, test_size=0.2, random_state=None):
    """Split dos dados em treino e teste"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def calcular_acuracia(y_true, y_pred):
    """Calcula a acurácia"""
    return np.mean(y_true == y_pred)

def simulacao_monte_carlo(X_mqo, Y_mqo, X_gauss, Y_gauss, y_labels, classes, R=500):
    """5. Simulação Monte Carlo para validação dos modelos"""
    print(f"\n=== 5. SIMULAÇÃO MONTE CARLO (R={R} rodadas) ===")
    
    lambdas = [0, 0.25, 0.5, 0.75, 1.0]
    modelos = [
        'MQO tradicional',
        'Classificador Gaussiano Tradicional',
        'Classificador Gaussiano (Cov. de todo cj. treino)',
        'Classificador Gaussiano (Cov. Agregada)',
        'Classificador de Bayes Ingênuo',
        'Classificador Gaussiano Regularizado (Friedman λ=0,25)',
        'Classificador Gaussiano Regularizado (Friedman λ=0,50)',
        'Classificador Gaussiano Regularizado (Friedman λ=0,75)',
        'Classificador Gaussiano Regularizado (Friedman λ=1,00)'
    ]
    
    acuracias = {modelo: [] for modelo in modelos}
    
    print("Executando simulações...")
    for r in range(R):
        if (r + 1) % 100 == 0:
            print(f"Rodada {r + 1}/{R}")
        
        # Particionamento 80% treino, 20% teste
        X_train, X_test, y_train, y_test = train_test_split_numpy(X_mqo, y_labels, test_size=0.2, random_state=r)
        
        # Criar Y_train para MQO
        Y_train = np.zeros((len(y_train), len(classes)))
        for i, classe in enumerate(classes):
            Y_train[y_train == classe, i] = 1
        
        # 1. MQO tradicional
        try:
            W = mqo_tradicional(X_train, Y_train)
            y_pred = predicao_mqo(X_test, W)
            # Converter predições para labels originais
            y_pred_labels = classes[y_pred]
            acc = calcular_acuracia(y_test, y_pred_labels)
            acuracias['MQO tradicional'].append(acc)
        except:
            acuracias['MQO tradicional'].append(0.0)
        
        # 2. Classificador Gaussiano Tradicional
        try:
            params = gaussiano_tradicional(X_train, Y_train)
            y_pred = predicao_gaussiana(X_test, params)
            y_pred_labels = classes[y_pred]
            acc = calcular_acuracia(y_test, y_pred_labels)
            acuracias['Classificador Gaussiano Tradicional'].append(acc)
        except:
            acuracias['Classificador Gaussiano Tradicional'].append(0.0)
        
        # 3. Classificador Gaussiano (Cov. de todo cj. treino)
        try:
            params = gaussiano_covariancias_iguais(X_train, Y_train)
            y_pred = predicao_gaussiana(X_test, params)
            y_pred_labels = classes[y_pred]
            acc = calcular_acuracia(y_test, y_pred_labels)
            acuracias['Classificador Gaussiano (Cov. de todo cj. treino)'].append(acc)
        except:
            acuracias['Classificador Gaussiano (Cov. de todo cj. treino)'].append(0.0)
        
        # 4. Classificador Gaussiano (Cov. Agregada)
        try:
            params = gaussiano_matriz_agregada(X_train, Y_train)
            y_pred = predicao_gaussiana(X_test, params)
            y_pred_labels = classes[y_pred]
            acc = calcular_acuracia(y_test, y_pred_labels)
            acuracias['Classificador Gaussiano (Cov. Agregada)'].append(acc)
        except:
            acuracias['Classificador Gaussiano (Cov. Agregada)'].append(0.0)
        
        # 5. Classificador de Bayes Ingênuo
        try:
            params = bayes_ingenuo(X_train, Y_train)
            y_pred = predicao_bayes_ingenuo(X_test, params)
            y_pred_labels = classes[y_pred]
            acc = calcular_acuracia(y_test, y_pred_labels)
            acuracias['Classificador de Bayes Ingênuo'].append(acc)
        except:
            acuracias['Classificador de Bayes Ingênuo'].append(0.0)
        
        # 6-9. Classificador Gaussiano Regularizado (Friedman)
        for lambda_val in lambdas[1:]:
            try:
                params = gaussiano_regularizado(X_train, Y_train, lambda_val)
                y_pred = predicao_gaussiana(X_test, params)
                y_pred_labels = classes[y_pred]
                acc = calcular_acuracia(y_test, y_pred_labels)
                # Usar a chave exata do dicionário
                chave = f'Classificador Gaussiano Regularizado (Friedman λ={lambda_val:.2f})'.replace('.', ',')
                acuracias[chave].append(acc)
            except:
                chave = f'Classificador Gaussiano Regularizado (Friedman λ={lambda_val:.2f})'.replace('.', ',')
                acuracias[chave].append(0.0)
    
    return acuracias

def analisar_resultados(acuracias):
    """6. Análise dos resultados"""
    print("\n=== 6. ANÁLISE DOS RESULTADOS ===")
    
    # Calcular estatísticas
    estatisticas = {}
    for modelo, acc_list in acuracias.items():
        acc_array = np.array(acc_list)
        estatisticas[modelo] = {
            'Média': np.mean(acc_array),
            'Desvio Padrão': np.std(acc_array),
            'Valor Mínimo': np.min(acc_array),
            'Valor Máximo': np.max(acc_array)
        }
    
    # Tabela no formato do trabalho
    print("\nTABELA DE RESULTADOS (ACURÁCIA):")
    print("=" * 120)
    print(f"{'Modelos':<60} {'Média':<15} {'Desvio-Padrão':<15} {'Maior Valor':<15} {'Menor Valor':<15}")
    print("=" * 120)
    
    modelos_ordem = [
        'MQO tradicional',
        'Classificador Gaussiano Tradicional',
        'Classificador Gaussiano (Cov. de todo cj. treino)',
        'Classificador Gaussiano (Cov. Agregada)',
        'Classificador de Bayes Ingênuo',
        'Classificador Gaussiano Regularizado (Friedman λ=0,25)',
        'Classificador Gaussiano Regularizado (Friedman λ=0,5)',
        'Classificador Gaussiano Regularizado (Friedman λ=0,75)',
        'Classificador Gaussiano Regularizado (Friedman λ=1,0)'
    ]
    
    for modelo in modelos_ordem:
        if modelo in estatisticas:
            stats = estatisticas[modelo]
            print(f"{modelo:<60} {stats['Média']:<15.4f} {stats['Desvio Padrão']:<15.4f} "
                  f"{stats['Valor Máximo']:<15.4f} {stats['Valor Mínimo']:<15.4f}")
    
    print("=" * 120)
    
    # Discussão dos resultados
    print("\n=== DISCUSSÃO DOS RESULTADOS ===")
    melhor_modelo = max(estatisticas.keys(), key=lambda x: estatisticas[x]['Média'])
    pior_modelo = min(estatisticas.keys(), key=lambda x: estatisticas[x]['Média'])
    
    print(f"• Melhor modelo: {melhor_modelo}")
    print(f"  - Acurácia Média: {estatisticas[melhor_modelo]['Média']:.4f}")
    print(f"• Pior modelo: {pior_modelo}")
    print(f"  - Acurácia Média: {estatisticas[pior_modelo]['Média']:.4f}")
    
    print(f"\n• Efeito da regularização:")
    gauss_trad = estatisticas['Classificador Gaussiano Tradicional']['Média']
    for lambda_val in [0.25, 0.5, 0.75, 1.0]:
        acc_reg = estatisticas[f'Classificador Gaussiano Regularizado (Friedman λ={lambda_val:.2f})'.replace('.', ',')]['Média']
        melhoria = ((acc_reg - gauss_trad) / gauss_trad) * 100
        print(f"  - λ={lambda_val}: {'Melhoria' if melhoria > 0 else 'Piora'} de {abs(melhoria):.1f}%")
    
    # Salvar resultados
    with open('resultados_classificacao_emg.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DA ANÁLISE DE CLASSIFICAÇÃO - EMG\n")
        f.write("=" * 120 + "\n\n")
        f.write("TABELA DE RESULTADOS (ACURÁCIA):\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Modelos':<60} {'Média':<15} {'Desvio-Padrão':<15} {'Maior Valor':<15} {'Menor Valor':<15}\n")
        f.write("=" * 120 + "\n")
        
        for modelo in modelos_ordem:
            if modelo in estatisticas:
                stats = estatisticas[modelo]
                f.write(f"{modelo:<60} {stats['Média']:<15.4f} {stats['Desvio Padrão']:<15.4f} "
                       f"{stats['Valor Máximo']:<15.4f} {stats['Valor Mínimo']:<15.4f}\n")
        
        f.write("=" * 120 + "\n")
        f.write(f"\nMelhor modelo: {melhor_modelo}\n")
        f.write(f"Pior modelo: {pior_modelo}\n")
    
    print(f"\n• Resultados salvos em 'resultados_classificacao_emg.txt'")
    
    return estatisticas

def main():
    """Função principal"""
    print("=== ANÁLISE DE CLASSIFICAÇÃO - EMG ===")
    
    # 1. Carregar dados
    data = carregar_dados()
    
    # 2. Organizar dados
    X_mqo, Y_mqo, X_gauss, Y_gauss, y_labels, classes = organizar_dados(data)
    
    # 3. Visualização inicial
    visualizacao_inicial(X_mqo, y_labels, classes)
    
    # 4-5. Simulação Monte Carlo
    acuracias = simulacao_monte_carlo(X_mqo, Y_mqo, X_gauss, Y_gauss, y_labels, classes, R=500)
    
    # 6. Análise dos resultados
    estatisticas = analisar_resultados(acuracias)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    return estatisticas

if __name__ == "__main__":
    resultados = main()

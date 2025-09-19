import numpy as np
import matplotlib.pyplot as plt

def carregar_dados():
    """Carrega os dados do aerogerador"""
    print("Carregando dados do aerogerador...")
    dados = np.loadtxt('data/aerogerador.dat', delimiter='\t')
    print(f"Dados carregados: {dados.shape[0]} observações")
    return dados

def visualizacao_inicial(dados):
    """1. Visualização inicial dos dados"""
    print("\n=== 1. VISUALIZAÇÃO INICIAL DOS DADOS ===")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(dados[:, 0], dados[:, 1], alpha=0.6, s=20)
    plt.xlabel('Velocidade do Vento (m/s)')
    plt.ylabel('Potência Gerada (kW)')
    plt.title('Relação entre Velocidade do Vento e Potência Gerada')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Estatísticas básicas
    print(f"Velocidade - Min: {np.min(dados[:, 0]):.2f}, Max: {np.max(dados[:, 0]):.2f}, Média: {np.mean(dados[:, 0]):.2f}")
    print(f"Potência - Min: {np.min(dados[:, 1]):.2f}, Max: {np.max(dados[:, 1]):.2f}, Média: {np.mean(dados[:, 1]):.2f}")
    print(f"Correlação: {np.corrcoef(dados[:, 0], dados[:, 1])[0, 1]:.3f}")

def organizar_dados(dados):
    """2. Organizar dados em matriz X e vetor y"""
    print("\n=== 2. ORGANIZAÇÃO DOS DADOS ===")
    
    X = dados[:, 0].reshape(-1, 1)  # Matriz X (variáveis regressoras)
    y = dados[:, 1]                 # Vetor y (variável dependente)
    
    print(f"Matriz X (variáveis regressoras): {X.shape}")
    print(f"Vetor y (variável dependente): {y.shape}")
    
    return X, y

def mqo_tradicional(X_train, y_train):
    """MQO tradicional com intercepto"""
    X_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train])
    try:
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_train
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_train
    return beta

def mqo_regularizado(X_train, y_train, lambda_reg):
    """MQO regularizado (Tikhonov/Ridge) com intercepto"""
    X_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train])
    I = np.eye(X_with_intercept.shape[1])
    I[0, 0] = 0  # Não regularizar o intercepto
    try:
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + lambda_reg * I) @ X_with_intercept.T @ y_train
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept + lambda_reg * I) @ X_with_intercept.T @ y_train
    return beta

def media_observaveis(y_train):
    """Modelo da média dos valores observáveis"""
    return np.array([[np.mean(y_train)], [0]])

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

def fazer_predicao(X_test, beta):
    """Faz predições"""
    X_test_with_intercept = np.column_stack([np.ones(X_test.shape[0]), X_test])
    return X_test_with_intercept @ beta

def calcular_rss(y_true, y_pred):
    """Calcula a soma dos desvios quadráticos (RSS)"""
    return np.sum((y_true - y_pred)**2)

def simulacao_monte_carlo(X, y, R=500):
    """3-6. Simulação Monte Carlo para validação dos modelos"""
    print(f"\n=== 3-6. SIMULAÇÃO MONTE CARLO (R={R} rodadas) ===")
    
    lambdas = [0, 0.25, 0.5, 0.75, 1.0]
    modelos = ['MQO Tradicional', 'MQO λ=0.25', 'MQO λ=0.5', 'MQO λ=0.75', 'MQO λ=1.0', 'Média Observáveis']
    
    rss_resultados = {modelo: [] for modelo in modelos}
    
    print("Executando simulações...")
    for r in range(R):
        if (r + 1) % 100 == 0:
            print(f"Rodada {r + 1}/{R}")
        
        # Particionamento 80% treino, 20% teste
        X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2, random_state=r)
        
        # 1. MQO Tradicional
        beta_mqo = mqo_tradicional(X_train, y_train)
        y_pred_mqo = fazer_predicao(X_test, beta_mqo)
        rss_mqo = calcular_rss(y_test, y_pred_mqo)
        rss_resultados['MQO Tradicional'].append(rss_mqo)
        
        # 2-5. MQO Regularizado para diferentes lambdas
        for lambda_val in lambdas[1:]:
            beta_reg = mqo_regularizado(X_train, y_train, lambda_val)
            y_pred_reg = fazer_predicao(X_test, beta_reg)
            rss_reg = calcular_rss(y_test, y_pred_reg)
            rss_resultados[f'MQO λ={lambda_val}'].append(rss_reg)
        
        # 6. Média dos valores observáveis
        beta_media = media_observaveis(y_train)
        y_pred_media = fazer_predicao(X_test, beta_media)
        rss_media = calcular_rss(y_test, y_pred_media)
        rss_resultados['Média Observáveis'].append(rss_media)
    
    return rss_resultados

def analisar_resultados(resultados):
    """Análise dos resultados"""
    print("\n=== ANÁLISE DOS RESULTADOS ===")
    
    # Calcular estatísticas
    estatisticas = {}
    for modelo, rss_list in resultados.items():
        rss_array = np.array(rss_list)
        estatisticas[modelo] = {
            'Média': np.mean(rss_array),
            'Desvio Padrão': np.std(rss_array),
            'Valor Mínimo': np.min(rss_array),
            'Valor Máximo': np.max(rss_array)
        }
    
    # Tabela no formato do trabalho
    print("\nTABELA DE RESULTADOS (RSS):")
    print("=" * 80)
    print(f"{'Modelos':<30} {'Média':<15} {'Desvio-Padrão':<15} {'Maior Valor':<15} {'Menor Valor':<15}")
    print("=" * 80)
    
    modelos_ordem = [
        'Média Observáveis',
        'MQO Tradicional', 
        'MQO λ=0.25',
        'MQO λ=0.5',
        'MQO λ=0.75',
        'MQO λ=1.0'
    ]
    
    for modelo in modelos_ordem:
        if modelo in estatisticas:
            stats = estatisticas[modelo]
            print(f"{modelo:<30} {stats['Média']:<15.2f} {stats['Desvio Padrão']:<15.2f} "
                  f"{stats['Valor Máximo']:<15.2f} {stats['Valor Mínimo']:<15.2f}")
    
    print("=" * 80)
    
    # Discussão dos resultados
    print("\n=== DISCUSSÃO DOS RESULTADOS ===")
    melhor_modelo = min(estatisticas.keys(), key=lambda x: estatisticas[x]['Média'])
    pior_modelo = max(estatisticas.keys(), key=lambda x: estatisticas[x]['Média'])
    
    print(f"• Melhor modelo: {melhor_modelo}")
    print(f"  - RSS Médio: {estatisticas[melhor_modelo]['Média']:.2f}")
    print(f"• Pior modelo: {pior_modelo}")
    print(f"  - RSS Médio: {estatisticas[pior_modelo]['Média']:.2f}")
    
    print(f"\n• Efeito da regularização:")
    mqo_trad = estatisticas['MQO Tradicional']['Média']
    for lambda_val in [0.25, 0.5, 0.75, 1.0]:
        rss_reg = estatisticas[f'MQO λ={lambda_val}']['Média']
        melhoria = ((mqo_trad - rss_reg) / mqo_trad) * 100
        print(f"  - λ={lambda_val}: {'Melhoria' if melhoria > 0 else 'Piora'} de {abs(melhoria):.1f}%")
    
    # Salvar resultados
    with open('resultados_regressao_aerogerador.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DA ANÁLISE DE REGRESSÃO - AEROGERADOR\n")
        f.write("=" * 80 + "\n\n")
        f.write("TABELA DE RESULTADOS (RSS):\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Modelos':<30} {'Média':<15} {'Desvio-Padrão':<15} {'Maior Valor':<15} {'Menor Valor':<15}\n")
        f.write("=" * 80 + "\n")
        
        for modelo in modelos_ordem:
            if modelo in estatisticas:
                stats = estatisticas[modelo]
                f.write(f"{modelo:<30} {stats['Média']:<15.2f} {stats['Desvio Padrão']:<15.2f} "
                       f"{stats['Valor Máximo']:<15.2f} {stats['Valor Mínimo']:<15.2f}\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"\nMelhor modelo: {melhor_modelo}\n")
        f.write(f"Pior modelo: {pior_modelo}\n")
    
    print(f"\n• Resultados salvos em 'resultados_regressao_aerogerador.txt'")
    
    return estatisticas

def main():
    """Função principal"""
    print("=== ANÁLISE DE REGRESSÃO - AEROGERADOR ===")
    
    # 1. Carregar dados
    dados = carregar_dados()
    
    # 2. Visualização inicial
    visualizacao_inicial(dados)
    
    # 3. Organizar dados
    X, y = organizar_dados(dados)
    
    # 4-6. Simulação Monte Carlo
    resultados = simulacao_monte_carlo(X, y, R=500)
    
    # 7. Análise dos resultados
    estatisticas = analisar_resultados(resultados)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    return estatisticas

if __name__ == "__main__":
    resultados = main()

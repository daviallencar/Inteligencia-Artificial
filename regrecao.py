import numpy as np
import matplotlib.pyplot as plt

class RegressaoAerogerador:
    def __init__(self):
        self.dados = None
        self.X = None
        self.y = None
        self.resultados = {}
        
    def carregar_dados(self, caminho_arquivo):
        """Carrega os dados do aerogerador usando apenas numpy"""
        print("Carregando dados do aerogerador...")
        
        # Ler arquivo usando numpy
        dados_raw = np.loadtxt(caminho_arquivo, delimiter='\t')
        
        # Organizar em dicionário para facilitar acesso
        self.dados = {
            'velocidade_vento': dados_raw[:, 0],
            'potencia': dados_raw[:, 1]
        }
        
        print(f"Dados carregados: {len(self.dados['velocidade_vento'])} observações")
        return self.dados
    
    def visualizacao_inicial(self):
        """1. Visualização inicial dos dados"""
        print("\n=== 1. VISUALIZAÇÃO INICIAL DOS DADOS ===")
        
        plt.figure(figsize=(12, 8))
        
        # Gráfico de espalhamento
        plt.subplot(2, 2, 1)
        plt.scatter(self.dados['velocidade_vento'], self.dados['potencia'], alpha=0.6, s=20)
        plt.xlabel('Velocidade do Vento (m/s)')
        plt.ylabel('Potência Gerada (kW)')
        plt.title('Relação entre Velocidade do Vento e Potência Gerada')
        plt.grid(True, alpha=0.3)
        
        # Histograma da velocidade do vento
        plt.subplot(2, 2, 2)
        plt.hist(self.dados['velocidade_vento'], bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('Velocidade do Vento (m/s)')
        plt.ylabel('Frequência')
        plt.title('Distribuição da Velocidade do Vento')
        plt.grid(True, alpha=0.3)
        
        # Histograma da potência
        plt.subplot(2, 2, 3)
        plt.hist(self.dados['potencia'], bins=30, alpha=0.7, color='lightcoral')
        plt.xlabel('Potência Gerada (kW)')
        plt.ylabel('Frequência')
        plt.title('Distribuição da Potência Gerada')
        plt.grid(True, alpha=0.3)
        
        # Estatísticas descritivas
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Calcular estatísticas usando numpy
        vel_media = np.mean(self.dados['velocidade_vento'])
        vel_std = np.std(self.dados['velocidade_vento'])
        vel_min = np.min(self.dados['velocidade_vento'])
        vel_max = np.max(self.dados['velocidade_vento'])
        
        pot_media = np.mean(self.dados['potencia'])
        pot_std = np.std(self.dados['potencia'])
        pot_min = np.min(self.dados['potencia'])
        pot_max = np.max(self.dados['potencia'])
        
        # Calcular correlação usando numpy
        correlacao = np.corrcoef(self.dados['velocidade_vento'], self.dados['potencia'])[0, 1]
        
        stats_text = f"""
        ESTATÍSTICAS DESCRITIVAS
        
        Velocidade do Vento:
        • Média: {vel_media:.2f} m/s
        • Desvio Padrão: {vel_std:.2f} m/s
        • Mínimo: {vel_min:.2f} m/s
        • Máximo: {vel_max:.2f} m/s
        
        Potência Gerada:
        • Média: {pot_media:.2f} kW
        • Desvio Padrão: {pot_std:.2f} kW
        • Mínimo: {pot_min:.2f} kW
        • Máximo: {pot_max:.2f} kW
        
        Correlação: {correlacao:.3f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
        
        # Discussão sobre características do modelo
        print("\nDISCUSSÃO SOBRE CARACTERÍSTICAS DO MODELO:")
        print("• Relação não-linear: A potência não cresce linearmente com a velocidade")
        print("• Curva de potência típica: Baixa potência em velocidades baixas, crescimento rápido")
        print("• Possível saturação: Em velocidades muito altas, a potência pode se estabilizar")
        print("• Modelo adequado: Regressão polinomial ou não-linear seria mais apropriada")
        print("• Regularização necessária: Para evitar overfitting em modelos complexos")
    
    def organizar_dados(self):
        """2. Organizar dados em matriz X e vetor y"""
        print("\n=== 2. ORGANIZAÇÃO DOS DADOS ===")
        
        # Criar matriz X (variáveis regressoras)
        self.X = self.dados['velocidade_vento'].reshape(-1, 1)  # Reshape para coluna
        self.y = self.dados['potencia']
        
        print(f"Matriz X (variáveis regressoras): {self.X.shape}")
        print(f"Vetor y (variável dependente): {self.y.shape}")
        print(f"Primeiras 5 observações de X:\n{self.X[:5].flatten()}")
        print(f"Primeiras 5 observações de y:\n{self.y[:5]}")
        
        return self.X, self.y
    
    def train_test_split_numpy(self, X, y, test_size=0.2, random_state=None):
        """Implementação do train_test_split usando apenas numpy"""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # Criar índices aleatórios
        indices = np.random.permutation(n_samples)
        
        # Dividir índices
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Dividir dados
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def mqo_tradicional(self, X_train, y_train):
        """MQO tradicional com intercepto"""
        # Adicionar coluna de 1s para o intercepto
        X_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train])
        
        # Fórmula: β = (X'X)^(-1)X'y
        try:
            beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_train
        except np.linalg.LinAlgError:
            # Se matriz singular, usar pseudo-inversa
            beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y_train
        
        return beta
    
    def mqo_regularizado(self, X_train, y_train, lambda_reg):
        """MQO regularizado (Tikhonov/Ridge) com intercepto"""
        # Adicionar coluna de 1s para o intercepto
        X_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train])
        
        # Matriz de identidade (não regularizar o intercepto)
        I = np.eye(X_with_intercept.shape[1])
        I[0, 0] = 0  # Não regularizar o intercepto
        
        # Fórmula: β = (X'X + λI)^(-1)X'y
        try:
            beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + lambda_reg * I) @ X_with_intercept.T @ y_train
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept + lambda_reg * I) @ X_with_intercept.T @ y_train
        
        return beta
    
    def media_observaveis(self, y_train):
        """Modelo da média dos valores observáveis (igual ao professor)"""
        # Criar vetor beta com intercepto = média e coeficientes = 0
        beta_media = np.array([[np.mean(y_train)], [0]])
        return beta_media
    
    def calcular_rss(self, y_true, y_pred):
        """Calcula a soma dos desvios quadráticos (RSS)"""
        return np.sum((y_true - y_pred)**2)
    
    def fazer_predicao(self, X_test, beta, modelo_tipo='mqo'):
        """Faz predições baseadas no modelo treinado"""
        # Adicionar intercepto
        X_test_with_intercept = np.column_stack([np.ones(X_test.shape[0]), X_test])
        return X_test_with_intercept @ beta
    
    def simulacao_monte_carlo(self, R=500):
        """3-6. Simulação Monte Carlo para validação dos modelos"""
        print(f"\n=== 3-6. SIMULAÇÃO MONTE CARLO (R={R} rodadas) ===")
        
        lambdas = [0, 0.25, 0.5, 0.75, 1.0]
        modelos = ['MQO Tradicional', 'MQO λ=0.25', 'MQO λ=0.5', 'MQO λ=0.75', 'MQO λ=1.0', 'Média Observáveis']
        
        # Listas para armazenar RSS de cada modelo
        rss_resultados = {modelo: [] for modelo in modelos}
        
        print("Executando simulações...")
        for r in range(R):
            if (r + 1) % 100 == 0:
                print(f"Rodada {r + 1}/{R}")
            
            # Particionamento 80% treino, 20% teste usando nossa implementação
            X_train, X_test, y_train, y_test = self.train_test_split_numpy(
                self.X, self.y, test_size=0.2, random_state=r
            )
            
            # 1. MQO Tradicional
            beta_mqo = self.mqo_tradicional(X_train, y_train)
            y_pred_mqo = self.fazer_predicao(X_test, beta_mqo)
            rss_mqo = self.calcular_rss(y_test, y_pred_mqo)
            rss_resultados['MQO Tradicional'].append(rss_mqo)
            
            # 2-5. MQO Regularizado para diferentes lambdas
            for i, lambda_val in enumerate(lambdas[1:], 1):  # Pular lambda=0 (já é MQO tradicional)
                beta_reg = self.mqo_regularizado(X_train, y_train, lambda_val)
                y_pred_reg = self.fazer_predicao(X_test, beta_reg)
                rss_reg = self.calcular_rss(y_test, y_pred_reg)
                rss_resultados[f'MQO λ={lambda_val}'].append(rss_reg)
            
            # 6. Média dos valores observáveis
            beta_media = self.media_observaveis(y_train)
            y_pred_media = self.fazer_predicao(X_test, beta_media)
            rss_media = self.calcular_rss(y_test, y_pred_media)
            rss_resultados['Média Observáveis'].append(rss_media)
        
        self.resultados = rss_resultados
        return rss_resultados
    
    def analisar_resultados(self):
        """Análise e visualização dos resultados"""
        print("\n=== ANÁLISE DOS RESULTADOS ===")
        
        # Calcular estatísticas para cada modelo usando numpy
        estatisticas = {}
        for modelo, rss_list in self.resultados.items():
            rss_array = np.array(rss_list)
            estatisticas[modelo] = {
                'Média': np.mean(rss_array),
                'Desvio Padrão': np.std(rss_array),
                'Valor Mínimo': np.min(rss_array),
                'Valor Máximo': np.max(rss_array),
                'Mediana': np.median(rss_array)
            }
        
        # Criar tabela no formato do trabalho
        print("\nTABELA DE RESULTADOS (RSS):")
        print("=" * 80)
        print(f"{'Modelos':<30} {'Média':<15} {'Desvio-Padrão':<15} {'Maior Valor':<15} {'Menor Valor':<15}")
        print("=" * 80)
        
        # Ordenar modelos conforme o trabalho
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
        
        # Visualizações melhoradas
        self.criar_visualizacoes_detalhadas(estatisticas)
        
        # Discussão dos resultados
        print("\n=== DISCUSSÃO DOS RESULTADOS ===")
        
        # Encontrar melhor e pior modelo
        melhor_modelo_nome = min(estatisticas.keys(), key=lambda x: estatisticas[x]['Média'])
        pior_modelo_nome = max(estatisticas.keys(), key=lambda x: estatisticas[x]['Média'])
        
        melhor_modelo = estatisticas[melhor_modelo_nome]
        pior_modelo = estatisticas[pior_modelo_nome]
        
        print(f"• Melhor modelo: {melhor_modelo_nome}")
        print(f"  - RSS Médio: {melhor_modelo['Média']:.2f}")
        print(f"  - Desvio Padrão: {melhor_modelo['Desvio Padrão']:.2f}")
        
        print(f"• Pior modelo: {pior_modelo_nome}")
        print(f"  - RSS Médio: {pior_modelo['Média']:.2f}")
        print(f"  - Desvio Padrão: {pior_modelo['Desvio Padrão']:.2f}")
        
        print(f"\n• Efeito da regularização:")
        mqo_trad = estatisticas['MQO Tradicional']['Média']
        for lambda_val in [0.25, 0.5, 0.75, 1.0]:
            rss_reg = estatisticas[f'MQO λ={lambda_val}']['Média']
            melhoria = ((mqo_trad - rss_reg) / mqo_trad) * 100
            print(f"  - λ={lambda_val}: {'Melhoria' if melhoria > 0 else 'Piora'} de {abs(melhoria):.1f}%")
        
        # Salvar resultados em arquivo de texto no formato do trabalho
        with open('resultados_regressao_aerogerador.txt', 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DA ANÁLISE DE REGRESSÃO - AEROGERADOR\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TABELA DE RESULTADOS (RSS):\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Modelos':<30} {'Média':<15} {'Desvio-Padrão':<15} {'Maior Valor':<15} {'Menor Valor':<15}\n")
            f.write("=" * 80 + "\n")
            
            # Ordenar modelos conforme o trabalho
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
                    f.write(f"{modelo:<30} {stats['Média']:<15.2f} {stats['Desvio Padrão']:<15.2f} "
                           f"{stats['Valor Máximo']:<15.2f} {stats['Valor Mínimo']:<15.2f}\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"\nMelhor modelo: {melhor_modelo_nome}\n")
            f.write(f"Pior modelo: {pior_modelo_nome}\n")
            
            f.write(f"\nEfeito da regularização:\n")
            mqo_trad = estatisticas['MQO Tradicional']['Média']
            for lambda_val in [0.25, 0.5, 0.75, 1.0]:
                rss_reg = estatisticas[f'MQO λ={lambda_val}']['Média']
                melhoria = ((mqo_trad - rss_reg) / mqo_trad) * 100
                f.write(f"  - λ={lambda_val}: {'Melhoria' if melhoria > 0 else 'Piora'} de {abs(melhoria):.1f}%\n")
        
        print(f"\n• Resultados salvos em 'resultados_regressao_aerogerador.txt'")
        
        return estatisticas
    
    def criar_visualizacoes_detalhadas(self, estatisticas):
        """Cria visualizações detalhadas com pontos e linhas de regressão"""
        
        # 1. Gráfico principal: Dados + Linhas de Regressão
        plt.figure(figsize=(20, 12))
        
        # Subplot 1: Dados originais com linha de regressão
        plt.subplot(2, 4, 1)
        plt.scatter(self.dados['velocidade_vento'], self.dados['potencia'], 
                   alpha=0.6, s=20, color='blue', label='Dados')
        
        # Calcular e plotar linha de regressão
        X_full = self.dados['velocidade_vento'].reshape(-1, 1)
        y_full = self.dados['potencia']
        X_full_intercept = np.column_stack([np.ones(X_full.shape[0]), X_full])
        beta_full = np.linalg.inv(X_full_intercept.T @ X_full_intercept) @ X_full_intercept.T @ y_full
        
        x_line = np.linspace(0, 15, 100)
        y_line = beta_full[0] + beta_full[1] * x_line
        plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regressão: y = {beta_full[0]:.1f} + {beta_full[1]:.1f}x')
        
        plt.xlabel('Velocidade do Vento (m/s)')
        plt.ylabel('Potência Gerada (kW)')
        plt.title('Dados + Linha de Regressão')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Boxplot dos RSS
        plt.subplot(2, 4, 2)
        modelos_ordem = ['Média Observáveis', 'MQO Tradicional', 'MQO λ=0.25', 'MQO λ=0.5', 'MQO λ=0.75', 'MQO λ=1.0']
        dados_boxplot = [self.resultados[modelo] for modelo in modelos_ordem if modelo in self.resultados]
        plt.boxplot(dados_boxplot, tick_labels=[m.replace('MQO λ=', 'λ=') for m in modelos_ordem if m in self.resultados])
        plt.xticks(rotation=45)
        plt.ylabel('RSS')
        plt.title('Distribuição dos RSS por Modelo')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Gráfico de barras - Média dos RSS
        plt.subplot(2, 4, 3)
        medias = [estatisticas[modelo]['Média'] for modelo in modelos_ordem if modelo in estatisticas]
        cores = ['red' if 'Média' in modelo else 'skyblue' for modelo in modelos_ordem if modelo in estatisticas]
        plt.bar(range(len(medias)), medias, color=cores)
        plt.xticks(range(len(medias)), [m.replace('MQO λ=', 'λ=') for m in modelos_ordem if m in estatisticas], rotation=45)
        plt.ylabel('RSS Médio')
        plt.title('RSS Médio por Modelo')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Efeito da Regularização
        plt.subplot(2, 4, 4)
        modelos_mqo = [m for m in modelos_ordem if 'MQO' in m and m in estatisticas]
        rss_mqo = [estatisticas[m]['Média'] for m in modelos_mqo]
        lambdas_vals = [0, 0.25, 0.5, 0.75, 1.0]
        plt.plot(lambdas_vals, rss_mqo, 'o-', linewidth=2, markersize=8, color='green')
        plt.xlabel('Valor de λ (Lambda)')
        plt.ylabel('RSS Médio')
        plt.title('Efeito da Regularização')
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Histograma dos RSS (sem a média para melhor visualização)
        plt.subplot(2, 4, 5)
        modelos_sem_media = [m for m in modelos_ordem if 'Média' not in m and m in estatisticas]
        for i, modelo in enumerate(modelos_sem_media):
            plt.hist(self.resultados[modelo], bins=30, alpha=0.6, 
                    label=modelo.replace('MQO λ=', 'λ='), color=plt.cm.viridis(i/len(modelos_sem_media)))
        plt.xlabel('RSS')
        plt.ylabel('Frequência')
        plt.title('Distribuição dos RSS (Modelos MQO)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Comparação Média vs MQO
        plt.subplot(2, 4, 6)
        rss_media = estatisticas['Média Observáveis']['Média']
        rss_mqo_trad = estatisticas['MQO Tradicional']['Média']
        plt.bar(['Média Observáveis', 'MQO Tradicional'], [rss_media, rss_mqo_trad], 
                color=['red', 'green'], alpha=0.7)
        plt.ylabel('RSS Médio')
        plt.title('Comparação: Média vs MQO')
        plt.yscale('log')  # Escala log para melhor visualização
        plt.grid(True, alpha=0.3)
        
        # Subplot 7: Desvio Padrão
        plt.subplot(2, 4, 7)
        desvios = [estatisticas[modelo]['Desvio Padrão'] for modelo in modelos_ordem if modelo in estatisticas]
        plt.bar(range(len(desvios)), desvios, color='orange', alpha=0.7)
        plt.xticks(range(len(desvios)), [m.replace('MQO λ=', 'λ=') for m in modelos_ordem if m in estatisticas], rotation=45)
        plt.ylabel('Desvio Padrão do RSS')
        plt.title('Variabilidade dos RSS')
        plt.grid(True, alpha=0.3)
        
        # Subplot 8: Ranking dos modelos
        plt.subplot(2, 4, 8)
        ranking_data = sorted(estatisticas.items(), key=lambda x: x[1]['Média'])
        ranking_modelos = [item[0].replace('MQO λ=', 'λ=') for item in ranking_data]
        ranking_medias = [item[1]['Média'] for item in ranking_data]
        
        cores_ranking = ['red' if 'Média' in modelo else 'skyblue' for modelo in ranking_modelos]
        plt.barh(range(len(ranking_modelos)), ranking_medias, color=cores_ranking, alpha=0.7)
        plt.yticks(range(len(ranking_modelos)), ranking_modelos)
        plt.xlabel('RSS Médio')
        plt.title('Ranking dos Modelos')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Gráfico adicional: Análise de Resíduos
        self.criar_grafico_residuos()
    
    def criar_grafico_residuos(self):
        """Cria gráfico de análise de resíduos"""
        plt.figure(figsize=(15, 5))
        
        # Usar uma amostra dos dados para análise de resíduos
        X_sample = self.X[:1000]  # Primeiros 1000 pontos
        y_sample = self.y[:1000]
        
        # Calcular regressão
        X_sample_intercept = np.column_stack([np.ones(X_sample.shape[0]), X_sample])
        beta_sample = np.linalg.inv(X_sample_intercept.T @ X_sample_intercept) @ X_sample_intercept.T @ y_sample
        
        # Predições
        y_pred = X_sample_intercept @ beta_sample
        residuos = y_sample - y_pred
        
        # Subplot 1: Resíduos vs Valores Ajustados
        plt.subplot(1, 3, 1)
        plt.scatter(y_pred, residuos, alpha=0.6, s=20)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Resíduos')
        plt.title('Resíduos vs Valores Ajustados')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Resíduos vs Velocidade do Vento
        plt.subplot(1, 3, 2)
        plt.scatter(X_sample.flatten(), residuos, alpha=0.6, s=20)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Velocidade do Vento (m/s)')
        plt.ylabel('Resíduos')
        plt.title('Resíduos vs Velocidade do Vento')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Histograma dos resíduos
        plt.subplot(1, 3, 3)
        plt.hist(residuos, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Resíduos')
        plt.ylabel('Frequência')
        plt.title('Distribuição dos Resíduos')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Função principal para executar a análise de regressão"""
    print("=== ANÁLISE DE REGRESSÃO - AEROGERADOR ===")
    print("Usando apenas NumPy e Matplotlib")
    
    # Inicializar classe
    regressao = RegressaoAerogerador()
    
    # 1. Carregar dados
    dados = regressao.carregar_dados('data/aerogerador.dat')
    
    # 2. Visualização inicial
    regressao.visualizacao_inicial()
    
    # 3. Organizar dados
    X, y = regressao.organizar_dados()
    
    # 4-6. Simulação Monte Carlo
    resultados = regressao.simulacao_monte_carlo(R=500)
    
    # 7. Análise dos resultados
    estatisticas = regressao.analisar_resultados()
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    return regressao, estatisticas

if __name__ == "__main__":
    regressao, resultados = main()
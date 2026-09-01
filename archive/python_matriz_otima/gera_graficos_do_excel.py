import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# --- Configurações ---
excel_filename = 'python\\outputs\\resultados_simulacao_sdre_discreto.xlsx'
output_dir = 'python\\outputs\\'

print(f"Lendo arquivo Excel: {excel_filename}")

# --- Leitura do arquivo Excel ---
try:
    df = pd.DataFrame(pd.read_excel(excel_filename, sheet_name='Resultados'))
    print(f"Dados carregados com sucesso! Total de simulações: {len(df)}")
except Exception as e:
    print(f"Erro ao ler o arquivo Excel: {e}")
    exit(1)

# Verificar se o DataFrame tem dados
if df.empty:
    print("O arquivo Excel está vazio!")
    exit(1)

# Exibir primeiras linhas para verificação
print("\nPrimeiras linhas do DataFrame:")
print(df.head())

# --- Função para converter alphas em cores RGB ---
def alphas_to_rgb(alpha1, alpha2, alpha3, min_alpha, max_alpha):
    """
    Converte valores de alpha em cores RGB
    alpha1 -> R (vermelho)
    alpha2 -> G (verde)
    alpha3 -> B (azul)
    """
    # Normaliza os valores de alpha para o intervalo [0, 1]
    r = (alpha1 - min_alpha) / (max_alpha - min_alpha)
    g = (alpha2 - min_alpha) / (max_alpha - min_alpha)
    b = (alpha3 - min_alpha) / (max_alpha - min_alpha)
    return (r, g, b)

# --- Preparação dos dados ---
print("\n--- Processando dados para geração de gráficos ---")

# Calcula os valores mínimos e máximos de alpha
min_alpha = min(df['param1_alpha1'].min(), df['param2_alpha2'].min(), df['param3_alpha3'].min())
max_alpha = max(df['param1_alpha1'].max(), df['param2_alpha2'].max(), df['param3_alpha3'].max())

print(f"Intervalo de alphas: [{min_alpha:.2f}, {max_alpha:.2f}]")

# Gera cores RGB para cada combinação de parâmetros
cores_rgb = [alphas_to_rgb(row['param1_alpha1'], row['param2_alpha2'], row['param3_alpha3'], 
                            min_alpha, max_alpha) 
             for _, row in df.iterrows()]

# Extrai arrays de dados
custos = df['custo_total'].values
autovalores = df['maior_autovalor_abs'].values
indices = np.arange(len(custos))

# Estatísticas
custo_medio = np.mean(custos)
custo_mediana = np.median(custos)
autovalor_medio = np.mean(autovalores)
n_estaveis = sum(1 for av in autovalores if av < 1.0)
n_total = len(autovalores)

print(f"\nEstatísticas:")
print(f"  Custo médio: {custo_medio:.6f}")
print(f"  Custo mediana: {custo_mediana:.6f}")
print(f"  Autovalor médio: {autovalor_medio:.4f}")
print(f"  Sistemas estáveis: {n_estaveis}/{n_total} ({100*n_estaveis/n_total:.1f}%)")

# --- Configurar estilo dos gráficos ---
plt.style.use('seaborn-v0_8-whitegrid')

# Elementos da legenda RGB
legend_elements = [
    Rectangle((0, 0), 1, 1, fc=(1, 0, 0), label=f'α1: {min_alpha:.0f} (preto) → {max_alpha:.0f} (vermelho)'),
    Rectangle((0, 0), 1, 1, fc=(0, 1, 0), label=f'α2: {min_alpha:.0f} (preto) → {max_alpha:.0f} (verde)'),
    Rectangle((0, 0), 1, 1, fc=(0, 0, 1), label=f'α3: {min_alpha:.0f} (preto) → {max_alpha:.0f} (azul)')
]
'''
# --- GRÁFICO 1: Distribuição dos Custos ---
print("\n--- Gerando Gráfico 1: Distribuição dos Custos ---")
fig_custos, ax_custos = plt.subplots(figsize=(14, 7))

ax_custos.bar(indices, custos, color=cores_rgb, alpha=0.8, edgecolor='black', linewidth=0.5)
ax_custos.set_xlabel('Índice da Simulação (ordenado por custo)', fontsize=12)
ax_custos.set_ylabel('Custo Total (J)', fontsize=12)
ax_custos.set_title('Distribuição dos Custos Totais para Diferentes Combinações de Parâmetros\n(Cores: R=α1, G=α2, B=α3)', 
                    fontsize=14, fontweight='bold')
ax_custos.grid(True, alpha=0.3)

# Linha horizontal para o custo médio
ax_custos.axhline(y=custo_medio, color='orange', linestyle='--', linewidth=2, 
                  label=f'Custo Médio: {custo_medio:.6f}')

# Legenda
ax_custos.legend(handles=legend_elements + [Rectangle((0, 0), 1, 1, fc='orange', label=f'Custo Médio: {custo_medio:.6f}')], 
                loc='upper right', fontsize=9, title='Mapeamento RGB')

plt.tight_layout()
plt.savefig(output_dir + 'distribuicao_custos.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}distribuicao_custos.png")

# --- GRÁFICO 2: Distribuição dos Autovalores Máximos ---
print("--- Gerando Gráfico 2: Distribuição dos Autovalores ---")
fig_autovalores, ax_autovalores = plt.subplots(figsize=(14, 7))

ax_autovalores.bar(indices, autovalores, color=cores_rgb, alpha=0.8, edgecolor='black', linewidth=0.5)
ax_autovalores.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, label='Limite de Estabilidade (|λ| = 1)')
ax_autovalores.set_xlabel('Índice da Simulação (ordenado por custo)', fontsize=12)
ax_autovalores.set_ylabel('Maior Autovalor Absoluto (|λ|_max)', fontsize=12)
ax_autovalores.set_title('Distribuição dos Maiores Autovalores do Sistema em Malha Fechada\n(Cores: R=α1, G=α2, B=α3)', 
                        fontsize=14, fontweight='bold')
ax_autovalores.grid(True, alpha=0.3)

# Texto informativo
ax_autovalores.text(0.02, 0.98, f'Sistemas Estáveis: {n_estaveis}/{n_total} ({100*n_estaveis/n_total:.1f}%)',
                   transform=ax_autovalores.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Legenda
ax_autovalores.legend(handles=legend_elements + [Rectangle((0, 0), 1, 1, fc='red', label='Limite de Estabilidade')], 
                     loc='upper right', fontsize=9, title='Mapeamento RGB')

plt.tight_layout()
plt.savefig(output_dir + 'distribuicao_autovalores.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}distribuicao_autovalores.png")
'''
# --- GRÁFICO 3: Relação entre Custo e Autovalor Máximo ---
print("--- Gerando Gráfico 3: Relação Custo vs Autovalor ---")
fig_relacao, ax_relacao = plt.subplots(figsize=(11, 9))

# Pontos sem borda (apenas a cor de face) e com tamanho reduzido
# edgecolors='none' remove a borda preta; s controla o tamanho do ponto.
scatter = ax_relacao.scatter(autovalores, custos, c=cores_rgb,
                             s=5, alpha=0.8, edgecolors='none', linewidths=0)
ax_relacao.set_xlabel('Maior Autovalor Absoluto (|λ|_max)', fontsize=12)
ax_relacao.set_ylabel('Custo Total (J)', fontsize=12)
ax_relacao.set_title('Relação entre Custo Total e Estabilidade do Sistema\n(Cores: R=α1, G=α2, B=α3)', 
                     fontsize=14, fontweight='bold')
ax_relacao.axvline(x=1.0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Limite de Estabilidade')
ax_relacao.grid(True, alpha=0.3)

# Legenda
ax_relacao.legend(handles=legend_elements + [Rectangle((0, 0), 1, 1, fc='red', label='Limite de Estabilidade')], 
                 loc='best', fontsize=9, title='Mapeamento RGB')

plt.tight_layout()
plt.savefig(output_dir + 'relacao_custo_autovalor.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}relacao_custo_autovalor.png")

# --- GRÁFICO 4: Histograma de Custos com cores RGB ---
print("--- Gerando Gráfico 4: Histograma de Custos ---")
fig_hist_custo, ax_hist_custo = plt.subplots(figsize=(11, 7))

n_bins = 30
counts, bin_edges, patches = ax_hist_custo.hist(custos, bins=n_bins, edgecolor='black', linewidth=0.5)

# Colorir cada barra do histograma com a cor média das simulações naquele bin
for i, patch in enumerate(patches):
    mask = (custos >= bin_edges[i]) & (custos < bin_edges[i+1])
    if i == len(patches) - 1:  # Último bin inclui o valor máximo
        mask = (custos >= bin_edges[i]) & (custos <= bin_edges[i+1])
    
    if np.any(mask):
        cores_no_bin = [cores_rgb[j] for j in range(len(cores_rgb)) if mask[j]]
        if cores_no_bin:
            cor_media = np.mean(cores_no_bin, axis=0)
            patch.set_facecolor(cor_media)
            patch.set_alpha(0.7)

ax_hist_custo.axvline(x=custo_medio, color='red', linestyle='--', linewidth=2.5, 
                     label=f'Média: {custo_medio:.6f}')
ax_hist_custo.axvline(x=custo_mediana, color='blue', linestyle='--', linewidth=2.5, 
                     label=f'Mediana: {custo_mediana:.6f}')
ax_hist_custo.set_xlabel('Custo Total (J)', fontsize=12)
ax_hist_custo.set_ylabel('Frequência', fontsize=12)
ax_hist_custo.set_title('Histograma dos Custos Totais\n(Cores médias por bin: R=α1, G=α2, B=α3)', 
                       fontsize=14, fontweight='bold')
ax_hist_custo.legend(fontsize=10)
ax_hist_custo.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'histograma_custos.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}histograma_custos.png")

# --- GRÁFICO 5: Histograma de Autovalores com cores RGB ---
print("--- Gerando Gráfico 5: Histograma de Autovalores ---")
fig_hist_auto, ax_hist_auto = plt.subplots(figsize=(11, 7))

counts, bin_edges, patches = ax_hist_auto.hist(autovalores, bins=n_bins, edgecolor='black', linewidth=0.5)

# Colorir cada barra do histograma
for i, patch in enumerate(patches):
    mask = (autovalores >= bin_edges[i]) & (autovalores < bin_edges[i+1])
    if i == len(patches) - 1:
        mask = (autovalores >= bin_edges[i]) & (autovalores <= bin_edges[i+1])
    
    if np.any(mask):
        cores_no_bin = [cores_rgb[j] for j in range(len(cores_rgb)) if mask[j]]
        if cores_no_bin:
            cor_media = np.mean(cores_no_bin, axis=0)
            patch.set_facecolor(cor_media)
            patch.set_alpha(0.7)

ax_hist_auto.axvline(x=1.0, color='red', linestyle='--', linewidth=2.5, label='Limite de Estabilidade')
ax_hist_auto.axvline(x=autovalor_medio, color='blue', linestyle='--', linewidth=2.5, 
                    label=f'Média: {autovalor_medio:.4f}')
ax_hist_auto.set_xlabel('Maior Autovalor Absoluto (|λ|_max)', fontsize=12)
ax_hist_auto.set_ylabel('Frequência', fontsize=12)
ax_hist_auto.set_title('Histograma dos Maiores Autovalores\n(Cores médias por bin: R=α1, G=α2, B=α3)', 
                      fontsize=14, fontweight='bold')
ax_hist_auto.legend(fontsize=10)
ax_hist_auto.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'histograma_autovalores.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}histograma_autovalores.png")

# --- GRÁFICO 6: Visualização 3D da distribuição de parâmetros ---
print("--- Gerando Gráfico 6: Visualização 3D ---")
fig_3d = plt.figure(figsize=(12, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')

alpha1_vals = df['param1_alpha1'].values
alpha2_vals = df['param2_alpha2'].values
alpha3_vals = df['param3_alpha3'].values

scatter_3d = ax_3d.scatter(alpha1_vals, alpha2_vals, alpha3_vals, 
                           c=cores_rgb, s=100, alpha=0.7, 
                           edgecolors='black', linewidth=0.5)

ax_3d.set_xlabel('α1 (Vermelho)', fontsize=11, labelpad=10)
ax_3d.set_ylabel('α2 (Verde)', fontsize=11, labelpad=10)
ax_3d.set_zlabel('α3 (Azul)', fontsize=11, labelpad=10)
ax_3d.set_title('Distribuição 3D dos Parâmetros\n(Cores: R=α1, G=α2, B=α3)', 
                fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir + 'distribuicao_3d_parametros.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}distribuicao_3d_parametros.png")

# --- GRÁFICO 7: Mapa de calor 2D (Alpha1 vs Alpha2) ---
print("--- Gerando Gráfico 7: Mapa de Calor Alpha1 vs Alpha2 ---")
fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))

# Criar grade para o mapa de calor
scatter_heat = ax_heatmap.scatter(df['param1_alpha1'], df['param2_alpha2'], 
                                  c=custos, s=80, cmap='YlOrRd', 
                                  alpha=0.7, edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter_heat, ax=ax_heatmap)
cbar.set_label('Custo Total (J)', fontsize=11)

ax_heatmap.set_xlabel('α1', fontsize=12)
ax_heatmap.set_ylabel('α2', fontsize=12)
ax_heatmap.set_title('Mapa de Calor: Custo vs α1 e α2', fontsize=14, fontweight='bold')
ax_heatmap.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'mapa_calor_alpha1_alpha2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {output_dir}mapa_calor_alpha1_alpha2.png")

# --- RESUMO FINAL ---
print("\n" + "="*60)
print("RESUMO DOS RESULTADOS")
print("="*60)

# Encontrar melhor e pior configuração
idx_melhor = df['custo_total'].idxmin()
idx_pior = df['custo_total'].idxmax()

melhor = df.loc[idx_melhor]
pior = df.loc[idx_pior]

print(f"\n🏆 MELHOR CONFIGURAÇÃO:")
print(f"   α1 = {melhor['param1_alpha1']:.2f}")
print(f"   α2 = {melhor['param2_alpha2']:.2f}")
print(f"   α3 = {melhor['param3_alpha3']:.2f}")
print(f"   Custo = {melhor['custo_total']:.8f}")
print(f"   |λ|_max = {melhor['maior_autovalor_abs']:.6f}")
print(f"   Estável: {'Sim ✓' if melhor['maior_autovalor_abs'] < 1.0 else 'Não ✗'}")

print(f"\n❌ PIOR CONFIGURAÇÃO:")
print(f"   α1 = {pior['param1_alpha1']:.2f}")
print(f"   α2 = {pior['param2_alpha2']:.2f}")
print(f"   α3 = {pior['param3_alpha3']:.2f}")
print(f"   Custo = {pior['custo_total']:.8f}")
print(f"   |λ|_max = {pior['maior_autovalor_abs']:.6f}")
print(f"   Estável: {'Sim ✓' if pior['maior_autovalor_abs'] < 1.0 else 'Não ✗'}")

print("\n" + "="*60)
print(f"✅ Todos os {7} gráficos foram gerados com sucesso!")
print(f"📁 Localização: {output_dir}")
print("="*60)
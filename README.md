# 🔧 Sistema de Manutenção Preditiva com LSTM - App Streamlit

Um aplicativo Streamlit premium e elegante para análise e predição de falhas em máquinas industriais utilizando redes neurais LSTM (Long Short-Term Memory).

## 🎯 Características do App

### 🎨 Design Premium
- **Tema escuro elegante** com paleta de cores técnica (verde-azulado)
- **Interface responsiva** e moderna
- **Animações suaves** e transições
- **Visualizações interativas** com Plotly
- **Cards de métricas** com gradientes e efeitos visuais

### 📊 Funcionalidades Principais

#### 🏠 Página Inicial
- Visão geral do sistema com métricas principais
- Cards de status em tempo real
- Distribuição de falhas com gráficos interativos
- Informações técnicas do projeto

#### 📈 Análise de Dados
- Estatísticas descritivas das features
- Visualizações comparativas (normal vs falha)
- Matriz de correlação interativa
- Análise de features por estado

#### 🤖 Informações do Modelo
- Arquitetura detalhada da rede LSTM
- Parâmetros de configuração
- Resumo técnico do modelo
- Vantagens da arquitetura LSTM

#### 📊 Análise de Treinamento
- Gráficos de evolução (accuracy/loss)
- Métricas de convergência
- Análise de overfitting
- Insights e recomendações

#### 🔮 Interface de Predições
- **Predição Aleatória**: Teste com amostras do dataset
- **Predição Personalizada**: Interface para entrada manual de dados
- Gráficos de probabilidade com gauges
- Interpretação automática dos resultados

#### 💡 Insights Avançados
- **Análise de Performance**: Comparação com benchmarks
- **Padrões de Falhas**: Identificação de correlações críticas
- **Otimizações**: Sugestões de melhorias
- **Recomendações**: Roadmap para produção

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- TensorFlow 2.15+
- Streamlit 1.31+

### Instalação Rápida

1. **Clonar o repositório:**
```bash
git clone https://github.com/sidnei-almeida/manutencao_preditiva_lstm.git
cd manutencao_preditiva_lstm
```

2. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

3. **Executar o app:**
```bash
# Método 1: Script automático
./run_app.sh

# Método 2: Comando direto
streamlit run app_manutencao_preditiva.py --server.port 8502
```

4. **Acessar no navegador:**
```
http://localhost:8502
```

> **Nota**: O app carrega automaticamente os dados e modelo do repositório GitHub, não sendo necessário fazer download manual dos arquivos.

### Estrutura de Arquivos

O app carrega automaticamente os arquivos necessários do repositório GitHub:

```
manutencao_preditiva_lstm/
├── app_manutencao_preditiva.py    # App principal
├── .streamlit/
│   └── config.toml              # Configuração do Streamlit
├── requirements.txt              # Dependências
└── run_app.sh                   # Script de execução
```

**Arquivos carregados automaticamente do GitHub:**
- `dados/X_processed.npy` - Features processadas
- `dados/y_processed.npy` - Targets processados  
- `modelos/predictive_maintenance_model.keras` - Modelo treinado
- `treinamento/training_summary.json` - Histórico de treinamento

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.8+** - Linguagem principal
- **TensorFlow 2.15+** - Framework de Deep Learning
- **NumPy** - Computação numérica
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Pré-processamento e métricas

### Frontend
- **Streamlit** - Framework web para Python
- **Plotly** - Visualizações interativas
- **CSS3** - Estilização customizada
- **HTML5** - Estrutura da interface

### Machine Learning
- **LSTM (Long Short-Term Memory)** - Arquitetura de rede neural
- **Binary Classification** - Classificação binária
- **Time Series Analysis** - Análise de séries temporais
- **Predictive Maintenance** - Manutenção preditiva

## 🎨 Características Visuais

### Paleta de Cores
- **Primária**: `#00D4AA` (Verde-azulado técnico)
- **Secundária**: `#00B4D8` (Azul claro)
- **Accent**: `#0077B6` (Azul escuro)
- **Sucesso**: `#00D4AA` (Verde)
- **Aviso**: `#FFB347` (Laranja)
- **Erro**: `#FF6B6B` (Vermelho)

### Componentes Visuais
- **Cards de métricas** com gradientes e animações
- **Gráficos interativos** com Plotly
- **Indicadores de status** em tempo real
- **Formulários elegantes** para entrada de dados
- **Navegação lateral** com menu premium

## 📱 Responsividade

O app é totalmente responsivo e funciona perfeitamente em:
- 💻 **Desktop** (1920x1080+)
- 📱 **Tablet** (768x1024)
- 📱 **Mobile** (375x667+)

## 🔧 Personalização

### Modificar Cores
Edite as variáveis CSS no arquivo `app_manutencao_preditiva.py`:

```css
:root {
    --primary-color: #00D4AA;
    --secondary-color: #00B4D8;
    --accent-color: #0077B6;
    /* ... outras cores */
}
```

### Adicionar Novas Funcionalidades
1. Crie uma nova função seguindo o padrão `show_nova_funcionalidade()`
2. Adicione ao menu de navegação
3. Implemente a lógica na função `main()`

## 📊 Dados Suportados

O app espera dados no seguinte formato:
- **Features**: 7 colunas (temperatura, velocidade, torque, etc.)
- **Target**: Binário (0=Normal, 1=Falha)
- **Sequências**: 50 timesteps para LSTM

## 🎯 Métricas Exibidas

- **Acurácia**: Performance geral do modelo
- **Perda**: Função de custo durante treinamento
- **Precisão**: Estimativa baseada na acurácia
- **Recall**: Estimativa baseada na acurácia
- **Correlações**: Relação entre features e falhas

## 🚀 Próximos Passos

### Melhorias Planejadas
- [ ] Integração com APIs de sensores em tempo real
- [ ] Sistema de alertas por email/SMS
- [ ] Exportação de relatórios em PDF
- [ ] Dashboard de monitoramento contínuo
- [ ] Sistema de versionamento de modelos

### Deploy em Produção
- [ ] Containerização com Docker
- [ ] Deploy em cloud (AWS/Azure/GCP)
- [ ] CI/CD pipeline
- [ ] Monitoramento de performance
- [ ] Backup automático de modelos

## 🤝 Contribuição

Para contribuir com o projeto:
1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente as melhorias
4. Teste thoroughly
5. Submeta um Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👨‍💻 Autor

**Sidnei Almeida**
- GitHub: [@sidnei-almeida](https://github.com/sidnei-almeida)
- LinkedIn: [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)

---

## 🎉 Conclusão

Este app Streamlit representa uma solução completa e elegante para manutenção preditiva industrial, combinando:

- **Design premium** com UX/UI moderna
- **Funcionalidades avançadas** de análise e predição
- **Visualizações interativas** e informativas
- **Interface intuitiva** para usuários técnicos e não-técnicos
- **Arquitetura escalável** para futuras melhorias

O sistema está pronto para uso em ambientes industriais reais, oferecendo uma base sólida para implementação de manutenção preditiva inteligente.

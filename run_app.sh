#!/bin/bash

# Script para executar o app Streamlit de Manutenção Preditiva
# Sistema de Manutenção Preditiva com LSTM

echo "🔧 Iniciando Sistema de Manutenção Preditiva com LSTM..."
echo "📊 Carregando dados e modelo..."
echo "🚀 Iniciando servidor Streamlit..."

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    echo "🐍 Ativando ambiente virtual..."
    source venv/bin/activate
fi

# Verificar se os arquivos necessários existem
if [ ! -f "dados/X_processed.npy" ]; then
    echo "❌ Erro: Arquivo dados/X_processed.npy não encontrado!"
    exit 1
fi

if [ ! -f "dados/y_processed.npy" ]; then
    echo "❌ Erro: Arquivo dados/y_processed.npy não encontrado!"
    exit 1
fi

if [ ! -f "modelos/predictive_maintenance_model.keras" ]; then
    echo "❌ Erro: Arquivo modelos/predictive_maintenance_model.keras não encontrado!"
    exit 1
fi

if [ ! -f "treinamento/training_summary.json" ]; then
    echo "❌ Erro: Arquivo treinamento/training_summary.json não encontrado!"
    exit 1
fi

echo "✅ Todos os arquivos necessários encontrados!"
echo "🌐 Abrindo aplicação no navegador..."
echo "📱 Acesse: http://localhost:8502"

# Executar o app Streamlit
streamlit run app_manutencao_preditiva.py --server.port 8502 --server.address 0.0.0.0

# Analisador S21 - Docker

Aplicação para análise automática de ressonâncias em dados S21 usando Docker.

## Funcionalidades

- 📊 Upload de arquivos CSV via interface web
- 🔬 Detecção automática de ressonâncias
- 📈 Cálculo de fatores de qualidade e largura de banda
- 🎯 Cálculo de sensibilidade e figura de mérito
- 📥 Exportação de resultados em ZIP (Excel + TXT)
- 📱 Interface responsiva com Streamlit

## Como executar

### 1. Build e execução com Docker Compose (Recomendado)

```bash
# Build e execução
docker-compose up -d

# Acesse: http://localhost:8501
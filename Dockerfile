FROM python:3.13.4-slim-bookworm

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o aplicativo
COPY s21_analyzer_app.py .

# Expor a porta do Streamlit
EXPOSE 8501

# Saúde check para o container
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=30)"

# Comando para executar o aplicativo
ENTRYPOINT ["streamlit", "run", "s21_analyzer_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV QDRANT_COLLECTION_NAME=messages
ENV QDRANT_DENSE_VECTOR_NAME=dense
ENV QDRANT_SPARSE_VECTOR_NAME=sparse
ENV DENSE_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV DENSE_VECTOR_SIZE=384
ENV SPARSE_MODEL_NAME=Qdrant/bm25
ENV FASTEMBED_CACHE_PATH=/models/fastembed
ENV HF_HOME=/models/huggingface

RUN mkdir -p /models/fastembed /models/huggingface
RUN python -c "import os; from fastembed import SparseTextEmbedding, TextEmbedding; TextEmbedding(model_name=os.environ['DENSE_MODEL_NAME']); SparseTextEmbedding(model_name=os.environ['SPARSE_MODEL_NAME'])"

EXPOSE 8000

CMD ["python", "main.py"]

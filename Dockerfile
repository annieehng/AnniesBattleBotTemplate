FROM python:3.9-slim

RUN apt-get update && apt-get install -y default-jdk && \
    pip install --upgrade pip && \
    pip install requests pydantic tf-keras tensorflow==2.10.0 torch torchvision torchaudio openai language-tool-python "numpy<2.0" scikit-learn transformers pandas sentence-transformers xgboost

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
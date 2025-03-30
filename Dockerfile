FROM python:3.8

RUN apt-get update && apt-get install -y default-jdk && \
    pip install --upgrade pip && \
    pip install requests pydantic torch torchvision torchaudio openai language-tool-python numpy scikit-learn transformers

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]

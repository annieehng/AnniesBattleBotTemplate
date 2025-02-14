FROM python:3.8

RUN pip install --upgrade pip
RUN pip install requests
RUN pip install pydantic
RUN pip install tf-keras
RUN pip install tensorflow==2.10.0
RUN pip install torch torchvision torchaudio
RUN pip install openai language-tool-python numpy scikit-learn transformers

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]

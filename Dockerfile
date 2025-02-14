FROM python:3

RUN pip install requests
RUN pip install pydantic
RUN pip install tf-keras
RUN pip install tensorflow
RUN pip install torch torchvision torchaudio
RUN pip install openai language-tool-python numpy scikit-learn transformers

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]

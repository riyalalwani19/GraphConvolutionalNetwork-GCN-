FROM python:3.7

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html

RUN pip3 install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip


COPY ./ /app/

CMD ["python","main.py"]
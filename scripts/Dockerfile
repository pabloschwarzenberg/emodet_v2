FROM tensorflow/tensorflow:2.9.1-gpu

RUN mkdir scripts

WORKDIR /scripts

ADD . /scripts

RUN ./setup.sh

ENTRYPOINT ["./start.sh"]

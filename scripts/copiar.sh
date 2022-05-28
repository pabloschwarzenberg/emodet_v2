docker create -it --name dummy train1 bash
docker cp dummy:/scripts/modelo2/modelsG2.h5 .
docker cp dummy:/scripts/modelo2/weightG2.h5 .
docker rm -f dummy

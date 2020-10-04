#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "Specify container name"
    exit 1
fi

if [[ ! $(docker images da-ssd:dev | tail -1 | grep da-ssd) ]]
then
  docker build -t da-ssd:latest . -f docker/dev/Dockerfile --build-arg UID=$(id -u $(whoami)) --build-arg GID=$(id -g $(whoami))
fi

if [[ $(docker ps -f name=$1 | tail -1 | grep $1) ]]
then
    docker exec -it \
            --user dev \
            -e COLUMNS="`tput cols`" \
            -e LINES="`tput lines`" \
            $1 bash
elif [[ $(docker ps -f name=$1 -a | tail -1 | grep $1) ]]; then
    docker start $1
    docker exec -it --user dev -e COLUMNS="`tput cols`" -e LINES="`tput lines`" $1 bash
else
    docker run -it -d -P --ulimit memlock=-1 --ulimit stack=67108864 -v $(readlink -e ./):/home/dev/da-ssd \
      --name=$1 \
      --ipc=host \
      --gpus=all \
      -e COLUMNS="`tput cols`" \
      -e LINES="`tput lines`" \
      da-ssd:latest

    docker exec -it \
     --user dev \
     -e COLUMNS="`tput cols`" \
     -e LINES="`tput lines`" \
     $1 bash
fi

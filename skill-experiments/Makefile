.PHONY: build run

NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))

GPU ?= 0

GPUS = --gpus '"device=$(GPU)"'

# Get the current user ID and group ID
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

# Set flag for docker run command
# BASE_FLAGS=-it --rm -v ${PWD}:/home/myuser --shm-size 20G --user $(USER_ID):$(GROUP_ID)
BASE_FLAGS=-it --rm -v ${PWD}/../JaxMARL:/home/myuser/JaxMARL -v ${PWD}:/home/myuser/skill_ov2_experiments --shm-size 20G # --user $(USER_ID):$(GROUP_ID)

RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = overcooked-v2
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)

# make file commands
build:
	(cd .. && DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --tag $(IMAGE) -f ./experiments/Dockerfile --progress=plain .)

run:
	$(DOCKER_RUN) /bin/bash

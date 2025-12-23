FROM nvcr.io/nvidia/jax:23.10-py3

# install tmux
RUN apt-get update && \
    apt-get install -y tmux

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# RUN adduser -D myuser
# RUN useradd myuser -u 3632 -g 301 -m -s /bin/bash
RUN groupadd -g 301 mygroup && \
    useradd -m -u 3632 -g mygroup myuser

USER myuser
WORKDIR /home/myuser

ENV PATH="/home/myuser/.local/bin:$PATH"

# default workdir
# WORKDIR /home/workdir
# copy the JaxMARL folder to docker

COPY --chown=myuser:mygroup JaxMARL ./JaxMARL
RUN pip install --user -e JaxMARL

COPY --chown=myuser:mygroup experiments ./skill_ov2_experiments
RUN pip install --user -e skill_ov2_experiments

# RUN pip install --user -e skill_ov2_experiments/overcooked_ai

#for secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN git config --global --add safe.directory /home/myuser/JaxMARL
RUN git config --global --add safe.directory /home/myuser/skill_ov2_experiments

WORKDIR /home/myuser/skill_ov2_experiments


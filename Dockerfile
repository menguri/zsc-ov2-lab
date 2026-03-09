FROM nvcr.io/nvidia/jax:23.10-py3

# Select which experiment package to bake into the image.
# Examples:
#   OV2_VARIANT=experiments
#   OV2_VARIANT=experiments-stablock
#   OV2_VARIANT=experiments-discrete
ARG OV2_VARIANT=experiments

ARG USER_NAME=myuser
ARG USER_UID=3632
ARG USER_GID=301

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tmux \
      ffmpeg \
      libsm6 \
      libxext6 \
      git \
      openssh-client && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${USER_GID} ${USER_NAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USER_NAME}

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true

# Install local JaxMARL fork
COPY --chown=${USER_NAME}:${USER_GID} JaxMARL ./JaxMARL
RUN pip install --user -e JaxMARL

# Install selected experiment package
COPY --chown=${USER_NAME}:${USER_GID} ${OV2_VARIANT} ./overcooked_v2_experiments
RUN pip install --user -e overcooked_v2_experiments

# Keep useful defaults for wandb-authenticated runs.
ENV WANDB_API_KEY="" \
    WANDB_ENTITY=""

RUN git config --global --add safe.directory /home/${USER_NAME}/JaxMARL && \
    git config --global --add safe.directory /home/${USER_NAME}/overcooked_v2_experiments

WORKDIR /home/${USER_NAME}/overcooked_v2_experiments
CMD ["/bin/bash"]

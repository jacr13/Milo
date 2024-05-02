# --------- base ---------
FROM dmml/conda:ubt20.04-py310 AS base

WORKDIR /

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    xvfb unzip patchelf ffmpeg cmake swig git curl wget vim\
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install poetry && poetry self add poetry-dotenv-plugin

# --------- dev ---------
FROM base AS dev
COPY pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install -vvv --no-interaction --no-root 

# Set working directory and configure permissions for workspace
WORKDIR /workspace
RUN chmod -R a+w /workspace && git config --global --add safe.directory /workspace

# -------- final --------
FROM base AS final
COPY pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install -vvv --no-interaction --no-root --no-dev

# Set working directory and configure permissions for workspace
WORKDIR /workspace
RUN chmod -R a+w /workspace && git config --global --add safe.directory /workspace
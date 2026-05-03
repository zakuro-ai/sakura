FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV MKL_THREADING_LAYER=GNU
ENV UV_BREAK_SYSTEM_PACKAGES=1

# System packages: Python 3.12, OpenMPI, Redis, locale support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    sudo git libopenmpi-dev locales redis-server xz-utils curl \
    && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# Create user and directories
RUN useradd -m -s /bin/bash foo \
    && mkdir -p /opt/zakuro/pth /opt/zakuro/logs /opt/zakuro/bin \
    && chown -R foo:foo /opt/zakuro

# Install wheel (pulls all deps from pyproject.toml)
COPY dist/*.whl /tmp/
RUN uv pip install --system /tmp/*.whl && rm /tmp/*.whl

RUN ln -s $(which python3) /opt/zakuro/bin/sakura

# ---------------------------------------------------------------------------
# Test stage — installs wheel with all deps from pyproject.toml
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS test
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY dist/*.whl /tmp/
RUN uv pip install --system pytest /tmp/*.whl && rm /tmp/*.whl

COPY tests/ /workspace/tests/
COPY pyproject.toml /workspace/pyproject.toml
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import sakura" || exit 1
CMD ["python3", "-m", "pytest", "tests/", "-v"]

# ---------------------------------------------------------------------------
# Production stage — s6-overlay + redis service
# ---------------------------------------------------------------------------
FROM base AS production

# s6-overlay
ADD https://github.com/just-containers/s6-overlay/releases/download/v3.2.0.2/s6-overlay-noarch.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz
ADD https://github.com/just-containers/s6-overlay/releases/download/v3.2.0.2/s6-overlay-x86_64.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz

# s6 redis service
ARG TASK="redis"
WORKDIR /etc/s6-overlay/s6-rc.d/${TASK}/dependencies.d
COPY docker/static/${TASK}/run /etc/s6-overlay/s6-rc.d/${TASK}/run
RUN echo "longrun" > /etc/s6-overlay/s6-rc.d/${TASK}/type && \
    touch /etc/s6-overlay/s6-rc.d/user/contents.d/${TASK} \
    /etc/s6-overlay/s6-rc.d/${TASK}/dependencies.d/base

WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import sakura" && redis-cli ping | grep -q PONG || exit 1
ENTRYPOINT ["/init"]

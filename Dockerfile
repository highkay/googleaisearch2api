FROM mcr.microsoft.com/playwright:v1.58.2-noble

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH=/root/.local/bin:$PATH
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV BROWSER_WORKERS=1
ENV REQUEST_QUEUE_SIZE=8
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV NO_PROXY=${NO_PROXY}
ENV no_proxy=${no_proxy}

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN set -eux; \
    chromium_path="$(find /ms-playwright -path '*/chrome-linux64/chrome' -type f | head -n 1)"; \
    test -x "$chromium_path"; \
    mkdir -p /opt/google/chrome; \
    ln -sf "$chromium_path" /opt/google/chrome/chrome; \
    ln -sf "$chromium_path" /usr/bin/google-chrome; \
    ln -sf "$chromium_path" /usr/bin/google-chrome-stable

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY .env.example ./

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "googleaisearch2api"]

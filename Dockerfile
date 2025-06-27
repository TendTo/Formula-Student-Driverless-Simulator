FROM python:3.10-slim

# Install dependencies from apt
RUN apt update && apt install -y --no-install-recommends \
    git \
    wget \
    build-essential

COPY . /workspace/Formula-Student-Driverless-Simulator

WORKDIR /workspace/Formula-Student-Driverless-Simulator

RUN pip install .

ENTRYPOINT [ "fsds-ros" ]

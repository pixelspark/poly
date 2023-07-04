FROM node:18 AS clientbuilder
WORKDIR /src
COPY ./client .
RUN npm install && npm run build

FROM rust:1.70-bullseye as builder
# Install CUDA
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common \
	&& wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb \
	&& DEBIAN_FRONTEND=noninteractive dpkg -i cuda-keyring_1.0-1_all.deb \
	&& DEBIAN_FRONTEND=noninteractive add-apt-repository contrib \
	&& DEBIAN_FRONTEND=noninteractive apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-11-5

# Build llmd
WORKDIR /src

#RUN export
RUN export CUDA_PATH=/usr/local/cuda-11.5 && export PATH=$CUDA_PATH/bin:${PATH} && nvcc --version

COPY . .
RUN export CUDA_PATH=/usr/local/cuda-11.5 && export PATH=$CUDA_PATH/bin:${PATH} \
	&& cargo build --release --features=cublas --bin=llmd

#FROM debian:bullseye-slim
FROM nvidia/cuda:11.5.2-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates
WORKDIR /llmd
COPY --from=builder /src/target/release/llmd /llmd/llmd
#RUN cp /src/target/release/llmd /llmd/llmd
COPY --from=clientbuilder /src/dist /llmd/client/dist
CMD ["/llmd/llmd"]
HEALTHCHECK CMD curl --fail http://localhost/status || exit 1

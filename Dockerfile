FROM node:20 AS clientbuilder
WORKDIR /src
COPY ./client .
RUN npm install && npm run build

FROM rust:1.70 as builder
WORKDIR /src
COPY . .
RUN cargo build --release --bin=llmd

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates curl
WORKDIR /llmd
COPY --from=builder /src/target/release/llmd /llmd/llmd
COPY --from=clientbuilder /src/dist /llmd/client/dist
CMD ["/llmd/llmd"]
HEALTHCHECK CMD curl --fail http://localhost/status || exit 1
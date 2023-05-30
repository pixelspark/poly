FROM rust:1.67 as builder
WORKDIR /src
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates curl
WORKDIR /llmd
COPY --from=builder /src/target/release/llmd /llmd/llmd
COPY ./public /llmd/public
CMD ["/llmd/llmd"]
HEALTHCHECK CMD curl --fail http://localhost/status || exit 1
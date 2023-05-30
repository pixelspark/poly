FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates curl
WORKDIR /llmd
COPY ./target/release/llmd /llmd/llmd
COPY ./public /llmd/public
CMD ["/llmd/llmd"]
HEALTHCHECK CMD curl --fail http://localhost/status || exit 1
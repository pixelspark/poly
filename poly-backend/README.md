# Poly-backend

## Usage

### With Qdrant

```sh
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
curl -XPUT http://localhost:6333/collections/test -vvv -d '{"vectors":{"size":3200,"distance":"Cosine"}}' -H "Content-type: application/json"
```

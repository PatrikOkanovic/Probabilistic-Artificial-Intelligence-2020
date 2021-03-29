docker build --tag task2 .
docker run --memory="7g" --rm -v "$( cd "$( dirname "$0" )" && pwd )":/code task2

# Bayes network data generator
Mikołaj Garbowski

Application for generating data according to Bayes network representation of a random distribution

Assignment for Introduction to Artificial Intelligence Course @ Warsaw University of Technology

## Usage
Example network is represented in [network.json](./docs/network.json)

```shell
python main.py 10 -f ./docs/network.json > ./docs/data.csv
```

Produces [data.csv](./docs/data.csv)
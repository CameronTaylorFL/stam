# Unsupervised Progressive Learning and the STAM Architecture

This repository is the official implementation of [Unsupervised Progressive Learning and the STAM Architecture]

## Requirements
Intended for Ubuntu 18.04 and Python3-dev

To install Python 3 requirements:

```setup
sudo apt update
sudo apt install -y libsm6 libxext6 libxrender-dev
python3 -m venv .stam
source .stam/bin/activate
pip install -r requirements.txt
```

Download datasets from https://drive.google.com/file/d/1CLohFBp-uKiP35O_NtGvl9nPxnqlzijz/view?usp=sharing and unzip into this directory as datasets/

## Results

Our model achieves the following performance on :

### Classification
#### [Continual Image Classification on MNIST]

| Model name         | P1 Accuracy | P2 Accuracy | P3 Accuracy | P4 Accuracy | P5 Accuracy |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
|       STAM         |     97.6    |     97.4    |     96.2    |     95.0    |     91.2    |

#### [Continual Image Classification on SVHN]

| Model name         | P1 Accuracy | P2 Accuracy | P3 Accuracy | P4 Accuracy | P5 Accuracy |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
|       STAM         |     88.0    |     81.0    |     78.6    |     76.6    |     72.4    |

#### [Continual Image Classification on CIFAR-10]

| Model name         | P1 Accuracy | P2 Accuracy | P3 Accuracy | P4 Accuracy | P5 Accuracy |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
|       STAM         |     77.1    |     61.1    |     43.7    |     37.1    |     34.6    |

### Clustering

#### [Continual Image Clustering on MNIST]

| Model name         | P1 Accuracy | P2 Accuracy | P3 Accuracy | P4 Accuracy | P5 Accuracy |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
|       STAM         |     99.6    |     97.9    |     95.6    |     92.7    |     85.3    |

#### [Continual Image Clustering on SVHN]

| Model name         | P1 Accuracy | P2 Accuracy | P3 Accuracy | P4 Accuracy | P5 Accuracy |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
|       STAM         |     88.8    |     74.7    |     68.1    |     65.1    |     56.7    |

#### [Continual Image Clustering on CIFAR-10]

| Model name         | P1 Accuracy | P2 Accuracy | P3 Accuracy | P4 Accuracy | P5 Accuracy |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
|       STAM         |     65.5    |     51.1    |     38.8    |     33.5    |     30.4    |

## Contributing
MIT License
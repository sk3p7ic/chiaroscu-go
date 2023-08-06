package main

import "fmt"

import "github.com/sk3p7ic/chiaroscu-go/util"

const (
    DATASET_LOCAL_PATH = "datasets"
    DATASET_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    DATASET_TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
    DATASET_TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
    DATASET_TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
    DATASET_TEST_LABELS = "t10k-labels-idx1-ubyte.gz"
)

func main() {
    fmt.Println("Hello, World!")
    var filenames = []string{
        DATASET_TRAIN_IMAGES, DATASET_TRAIN_LABELS,
        DATASET_TEST_IMAGES, DATASET_TEST_LABELS,
    }
    util.LoadDataset(DATASET_LOCAL_PATH, DATASET_URL, filenames)
}

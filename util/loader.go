package util

import (
    "compress/gzip"
	"net/http"
    "bufio"
	"fmt"
	"io"
	"os"

	"github.com/schollz/progressbar/v3"
    "gonum.org/v1/gonum/mat"
)

// Number of dataset files to be checked
const N_DATASET_FILES = 4

// Dataset struct containing the dataset files
type Dataset struct {
    F_train_images os.File
    F_train_labels os.File
    F_test_images os.File
    F_test_labels os.File
}

// Dataset mode (train or test)
type DatasetMode uint8
// Dataset modes
const (
    dm_Train DatasetMode = iota
    dm_Test
)

// Parsed dataset fileset containing the labels and images as matrices
type ParsedFileset struct {
    Labels *mat.Dense
    Images *mat.Dense
}

// Parsed dataset containing the training and testing filesets as matrices
type ParsedDataset struct {
    Train ParsedFileset
    Test ParsedFileset
}

// Download a dataset file from a URL to a local path and filename.
func download_dataset_file(dataset_path, dataset_url, filename string) error {
	url := fmt.Sprintf("%s%s", dataset_url, filename)
	// Path to temporary file to be renamed after download
	var dwnld_file_path = fmt.Sprintf("%s/%s.dwnld", dataset_path, filename)
	// Create and send GET request
	req, _ := http.NewRequest("GET", url, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Create file and progress bar
	f, _ := os.Create(dwnld_file_path)
	defer f.Close()

	bar := progressbar.DefaultBytes(
		resp.ContentLength,
		fmt.Sprintf("    Retrieving file from '%s'...\n    ", url),
	)
	io.Copy(io.MultiWriter(f, bar), resp.Body)
	// Rename temporary file to actual filename
	os.Rename(dwnld_file_path, fmt.Sprintf("%s/%s", dataset_path, filename))

	return nil
}

// Check for a list of dataset files and download if not found. Returns a
// pointer to a Dataset struct containing the dataset files.
func LoadDataset(dataset_path, fallback_url string,
    filenames []string) *Dataset {
	// Verify that the number of filenames is correct
	if len(filenames) != N_DATASET_FILES {
		fmt.Println("[E] Invalid number of dataset files!")
		os.Exit(1)
	}
	fmt.Printf("Checking for datasets ")
	// Display list of datasets to be checked
	for i, dataset_name := range filenames {
		fmt.Printf("'%s/%s'", dataset_path, dataset_name)
		if i < len(filenames)-1 {
			fmt.Printf(", ")
		} else {
			fmt.Printf("...\n")
		}
	}
	// Check for each dataset file and download if not found
	for _, dataset_name := range filenames {
		_, err := os.Stat(fmt.Sprintf("%s/%s", dataset_path, dataset_name))
		if os.IsNotExist(err) {
			fmt.Println("[E] Dataset not found!")
			fmt.Printf("    Downloading dataset '%s'...\n", dataset_name)
			dl_err := download_dataset_file(dataset_path, fallback_url,
				dataset_name)
			if dl_err != nil {
				fmt.Println("[E] An error occurred while downloading the dataset.")
				fmt.Println(dl_err)
				os.Exit(1)
			}
		} else if err != nil {
			fmt.Println("[E] An error occurred while checking for the dataset.")
			fmt.Println(err)
			os.Exit(1)
		} else {
			fmt.Println("[I] Dataset found! Skipping download...")
		}
	}
    // Create array of dataset files
	var dataset_files = [N_DATASET_FILES]os.File{}
    // Open each dataset file
    fmt.Println("[I] Opening dataset files...")
	for i, dataset_name := range filenames {
		fmt.Printf("    LOAD '%s/%s'\n", dataset_path, dataset_name)
		f, _ := os.Open(fmt.Sprintf("%s/%s", dataset_path, dataset_name))
		dataset_files[i] = *f
	}
    // Return array of dataset files
	return &Dataset{ dataset_files[0], dataset_files[1],
        dataset_files[2], dataset_files[3] }
}

// Parse a dataset fileset into a ParsedFileset struct containing the labels
// and images as matrices.
func parse_dataset_fileset(dataset_image_file, dataset_label_file *os.File,
    mode DatasetMode) ParsedFileset {
    mode_str := func() string {
        if mode == dm_Train { return "training" } else { return "testing" }
    }()
    fmt.Printf("[I] Parsing %s dataset...\n", mode_str)
    gzip_reader_images, gri_err := gzip.NewReader(dataset_image_file)
    gzip_reader_labels, grl_err := gzip.NewReader(dataset_label_file)
    if gri_err != nil || grl_err != nil {
        fmt.Println("[E] An error occurred while parsing the dataset.")
        fmt.Println(gri_err)
        fmt.Println(grl_err)
        os.Exit(1)
    }
    defer gzip_reader_images.Close()
    defer gzip_reader_labels.Close()
    img_reader := bufio.NewReader(gzip_reader_images)
    lbl_reader := bufio.NewReader(gzip_reader_labels)
    // Read magic number, num images, num rows, and num columns from image file
    img_reader.Discard(16)
    // Read magic number and num labels from label file
    lbl_reader.Discard(8)
    // Get the number of images to be read
    n_images := func() uint32 {
        if mode == dm_Train { return 60000 } else { return 10000 }
    }()
    fmt.Printf("    Reading %d images...\n", n_images)
    var images = make([]float64, 784 * n_images)
    var labels = make([]float64, n_images)
    // Read image data
    for i := uint32(0); i < n_images; i++ {
        // Read image label
        label, _ := lbl_reader.ReadByte()
        labels[i] = float64(label)
        // Read image pixels
        for j := uint32(0); j < 784; j++ {
            pixel, _ := img_reader.ReadByte()
            images[i * 784 + j] = float64(pixel)
        }
    }
    fmt.Printf("    Finished reading %d images.\n", n_images)
    fmt.Println("    Assigning dataset into Dense matrices...")
    // Assign labels to Dense matrix
    labels_mat := mat.NewDense(int(n_images), 1, labels)
    images_mat := mat.NewDense(int(n_images), 784, images)
    fmt.Println("    Finished assigning dataset into Dense matrices.")
    return ParsedFileset{ labels_mat, images_mat }
}

// Parse a Dataset struct into a ParsedDataset struct containing the training
// and testing filesets as ParsedFileset structs, respectively.
func ParseDataset(dataset *Dataset) *ParsedDataset {
    train := parse_dataset_fileset(&dataset.F_train_images,
        &dataset.F_train_labels, dm_Train)
    test := parse_dataset_fileset(&dataset.F_test_images,
        &dataset.F_test_labels, dm_Test)
    // Print dataset information
    {
        fmt.Printf("[I] Parsed Training dataset into\n")
        r, c := train.Labels.Dims()
        fmt.Printf("    Label Vector: %d x %d\n", r, c)
        r, c = train.Images.Dims()
        fmt.Printf("    Image Matrix: %d x %d\n", r, c)

        fmt.Printf("[I] Parsed Testing dataset into\n")
        r, c = test.Labels.Dims()
        fmt.Printf("    Label Vector: %d x %d\n", r, c)
        r, c = test.Images.Dims()
        fmt.Printf("    Image Matrix: %d x %d\n", r, c)
    }
    return &ParsedDataset{ train, test }
}

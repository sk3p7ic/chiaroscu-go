package util

import (
	"fmt"
	"github.com/schollz/progressbar/v3"
	"io"
	"net/http"
	"os"
)

// Number of dataset files to be checked
const N_DATASET_FILES = 4

type Dataset struct {
    F_train_images os.File
    F_train_labels os.File
    F_test_images os.File
    F_test_labels os.File
}

// Download a dataset file from a URL to a local path and filename.
func download_dataset_file(dataset_path, dataset_url, filename string) error {
	url := fmt.Sprintf("%s/%s", dataset_url, filename)
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

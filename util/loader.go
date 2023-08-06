package util

import (
	"fmt"
	"os"
    "io"
    "net/http"
    "github.com/schollz/progressbar/v3"
)

func download_dataset_file(dataset_path, dataset_url, filename string) error {
    fmt.Println("Downloading dataset...")
    url := fmt.Sprintf("%s/%s", dataset_url, filename)
    var dwnld_file_path = fmt.Sprintf("%s/%s.dwnld", dataset_path, filename)
    req, _ := http.NewRequest("GET", url, nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    f, _ := os.Create(dwnld_file_path)
    
    bar := progressbar.DefaultBytes(
        resp.ContentLength,
        fmt.Sprintf("Retrieving file from '%s'...\n\t", url),
        )
    io.Copy(io.MultiWriter(f, bar), resp.Body)
    os.Rename(dwnld_file_path, fmt.Sprintf("%s/%s", dataset_path, filename))

    return nil
}

func LoadDataset(dataset_path, fallback_url string, filenames []string) {
	fmt.Printf("Checking for datasets ")
    for i, dataset_name := range filenames {
        fmt.Printf("'%s/%s'", dataset_path, dataset_name)
        if i < len(filenames) - 1 {
            fmt.Printf(", ")
        } else {
            fmt.Printf("...\n")
        }
    }
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
}

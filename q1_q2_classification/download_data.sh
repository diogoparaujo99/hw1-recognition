# This script downloads and extracts data to a specified folder 

data_folder="$(cd "$(dirname "$0")/../.." && pwd)/data"

mkdir -p "$data_folder"
cd "$data_folder"

echo "> Downloading datasets to $data_folder"

wget -nc https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar

wget -nc https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

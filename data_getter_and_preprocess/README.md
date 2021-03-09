To download the image i used, enter those command:
```
mkdir train test val

python downloader.py ids_train --download_folder=train --num_processes=42

python downloader.py ids_test --download_folder=test --num_processes=42

python downloader.py ids_val --download_folder=validation --num_processes=42
```
then download all the boxes file at this adress:
https://storage.googleapis.com/openimages/web/download.html

put it on a directory named data in the repo

and use clean_data_OP.ipynb to clean your data
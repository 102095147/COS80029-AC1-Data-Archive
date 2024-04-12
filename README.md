# COS80029-AC1-Data-Archive - Generate OpenNRE training dataset using GPT4

Install Python requirements using:
```sh
 pip install -r requirements.txt
```

Place raw text data to create a training dataset from in the ```inputs/``` folder and run the following to generate the JSONL dataset:
```sh
 python re_dataset_generation.py 
```

This will generate datasets for each raw text file under ```outputs/```.
To create the training, validation and testing datasets from the generated datasets in outputs, run the following:
```sh
 python create_train_test_val_datasets.py
```

This will create datasets in ```datasets/``` that are ready for training a model via OpenNRE.

# Create Your Own Image Classifier

This project is about building a flower classifier as part of Udacity's AI Programming with Python Nanodegree, using PyTorch.

The entire workflow is contained within a Jupyter Notebook.

## Python Scripts

There are two Python scripts provided:

- **`train.py`**: [Train the Classifier](https://github.com/bsassoli/Create-your-own-image-classifier/blob/master/train.py)
- **`predict.py`**: [Use the Classifier](https://github.com/bsassoli/Create-your-own-image-classifier/blob/master/predict.py)

### Training the Classifier

`train.py` is used to train the classifier. The user must provide one mandatory argument:

- `data_dir`: The path to the training data directory (str).

Optional arguments:

- `--save_dir`: Directory to save the trained model.
- `--arch`: Choose the neural network architecture. The default is VGG16, but densenet121 can also be specified.
- `--learning_r`: Set the learning rate for gradient descent. The default is 0.001.
- `--hidden_units`: Specify the number of neurons in an additional hidden layer (int).
- `--epochs`: Number of epochs (int). The default is 5.
- `--GPU`: Specify GPU if available; otherwise, the model will use the CPU.

### Using the Classifier

`predict.py` is used to classify images and output a probability ranking of predicted flower species. The only mandatory argument is:

- `--image_dir`: The path to the input image.

Optional arguments:

- `--load_dir`: Path to the model checkpoint.
- `--top_k`: Specify the number of top K-classes to output. The default is 5.
- `--category_names`: Path to a JSON file mapping categories to names.
- `--GPU`: Specify GPU if available; otherwise, the model will use the CPU.

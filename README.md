## Malaria Screening

Implementating of a deep neural network for Malary detection
Inspiration taken from:
https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/
###Dataset
2 classe:
- 13,794 images of parasitized
- 13,794 images of uninfected

With a total of 27,588 images.
You can acess the folder named 'cell_images.zip' at:
https://lhncbc.nlm.nih.gov/publication/pub9932

### Prerequisites
- Tensorflow2.0

### Run the models

**1)** After downloading the dataset, run the script **prep_data.py** to randomly separate the data into train and test sets. The variable `train_size` should be between 0 and 1 and it defines the proportion of the dataset that goes into the train set.

**2)** In the script **parameters.py** you can define the parameters to use while training the model. Such parameters are:
- `train_dir`: directory of the train set
- `val_dir`: directory of the validation set
- `input`: Input size of the images
- `batch_size`: batch size
- `epochs`: number of epochs

**3)** Run the script **train.py** to start training the model.

### Additional Scripts

- **generator.py**: Customized generator that yields batches of data 

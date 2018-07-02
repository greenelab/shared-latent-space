# Shared Latent Space Variational Autoencoders

## Motivation for application of Shared Latent SPace VAE's In Biology

Variational Autoencoders are machine learning models which learn the distribution of the data in a latent space manifold.
This allows the model to be trained on unsuprivesed data, but learning to recreate the input data, and also allows the model to
create new data that ressembles the original data by picking points in the lantent space manifold.
Shared Latent Space VAE's find relationships between two different domains and allow for transformations between the two.
They achieve this by linking the lantent space manifold between two different encoders and decoders.
This is particularly useful in Biology where we could use different data types as different 'views'
on the same biological problem.
The ability to transform between domains also allows us to transition between different data types.

## Diagram of Model
![Alt text](Shared_Latent_Space_VAE.png)

## Usage

### Computational Environment

All libraries and packages are handled by conda and specified in the `environment.yml` file.
To build and activate this enviroment, run:
```
#conda version 4.4.10
conda env create --force --file environment.yml

conda activate shared-latenet-space
```

### Running the Model

There are zipped pickle files with the MNIST, ICVL, and Cognoma data. The model will automatically unzip them.
The parameters of the model are passed in the command line argument.
You can consider changing the layout of the model in `shared_vae_class.py`
You can also control noise, dropout level, and warm start variables from command line arguments.

### Changing Datasets

If the data set is one of the already included datasets, you need to pass the name of that model in command line arguments.
Additionally, you should consider changing the model layout, as outlined in Running the Model.

### Adding More Datasets

If you want to add more Datasets, that is fully supported.
Create your own implementation of the `DataSetInfoAbstract` abstract class including a method for loading and visualizing.
You must return two training sets and two testing sets.
You do not have to return anything for visualization, so it is merely sufficent to define the function and return nothing.
Thus, you don't need to implement a visualization, but you must declare it to comply with the interface. 
When adding your own dataset, to call it from command line, you must add it to the `data_dict` in `main_file.py`.

## Files

### `main_file.py`

This is the file which should be called.
It handles calling other files for loading and formating data.
It also calls upon shared_vae_class.py to create the model, train it, and generate data.
As work continues, this file will become more general and easier to work with.
As of now, if you are using your own data, you should create a file for it which impliments the `DataSetInfoAbstract` abtract class.
Then add it to the `data_dict` dictionary so the program can link it with the command line parameter.

To run the file, open command line and enter:
```
python main_file.py --data --batchSize --numEpochs --firstLayerSizeLeft --thirdLayerSize --secondLayerSize   --encodedSize --firstLayerSizeRight --kappa --beta --noise --dropout --notes
```
Command Line Arguments:  
`--data`: the name of the dataset to use.  
`--batchSize`: the batch size to use for training.  
`--numEpochs`: number of epochs to train the model.  
`--firstLayerSizeLeft`: number of nodes in the first hidden layer of the left encoder.  
`--thirdLayerSize`: number of nodes in the third hidden layer, which is shared between the encoders.  
`--secondLayerSize`: number of nodes in the second hidden layer, which is shared between the encoders.  
`--encodedSize`: number of nodes in the shared latent space.  
`--firstLayerSizeRight`: number of nodes in the first hyidden layer of the right encoder.  
`--kappa`: the kappa variable for warm start, how much the beta ramps up each epoch.  
`--beta`: the beta variable for warm start.  
`--noise`: the amount of noise to apply to the training data.  
`--dropout`: the dropout for the model  
`--notes`: a string which will be saved with the parameters for reference notes.  
  
The `thirdLayerSize` and `secondLayerSize` parameters are for layers that are currently commented out.
### `shared_vae_class.py`

This file is the main class which hold the model.
It contains functions to compile, train, and generate from the model.
The model will take in a series of parameters which control size of layers, etc.
The model right now is very rigid in structure, but this may change.
There a 5 different models built inside here for the purposes of training, but they are hidden.
The generate function calls on the visualize function of the `DataSetInfoAbstract` class.
This will produce an image to help visualize how the model is working.
There are multiple layers in the model which are commented out, but can be reintroduced if a different
architecture is desired.

### `model_objects.py`

This file contains the `model_parameters` class which is fed to the `shared_vae_class` when it is initialized.
It also contains the `model` class which holds each of the models defined in the `shared_vae_class`.

### `DataSetInfoAbstractClass.py`

This file is an abstract class which is used for any dataset specific functions such as loading and visualizing.
These are abstract functions, so a specific implimentation must be provided for the dataset.

### `ICVL.py`

This is a specific implimentation of the `DataSetInfoAbstract` abstract class for the ICVL data.
It contains a load function which loads the data from a pickle file as well as a visualize function which produces images of the depth maps and knuckle maps.
The draw_hands function draws all of the lines between the joints in the hand as given by the dataset. 

### `MNIST.py`

This is a specific implimentation of the `DataSetInfoAbstract` abstract class for the MNIST data.
It contains a load function which loads the data from a pickle file as well as a visualize function which produces images of the regular MNIST digits and the inverse MNIST digits.

### `Cognoma.py`

This is a specific implimentation of the `DataSetInfoAbstract` abstract class for the Cognoma data.
It contains a load function which loads the data from a pickle file as well as a visualize function which produces many images and graphs.

## Output

The model saves various graphs and output into the specific folder for your data inside the `Output` folder. The model parameters will be saved in an `.html` with the number of the run serving as the title. All output will have the number of the run in the title. Additionally, diagrams of the left and right encoders will be saved, as will a line graph showing the training and validation loss over training. All other output will be determined by the `visualisation` method of it's data specific implementation of `DataSetInfoAbstract`.
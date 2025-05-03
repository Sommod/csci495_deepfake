# The GenConViT Model
There are three parts to this model: **MAE** (Masked AutoEncoder), **VAE** (Variational Encoder), and the combination of a **ConvNeXT** hybrid architect.
The model is mostly hard-coded to run a specific way, but can still run the testing results based on the input of the files.

# Instructions
## Dataset location
The Data needs to be loaded into the *data* folder with the parent directory being named *real_vs_fake*. Additionally, the next sub-directories should contain the *train, test, valid* directories containing the images.
CSV files are used for labels and information for the images, these need to be loaded into the **Util/Labels** directory. By default the program is using the 140k real or fake faces dataset and aleady contains the
CSV files for the dataset.

## Running the program
To run the program, either use the command:
>python src/Main.py
>

Additionally, you can run the **makefile** that is also present within the repository.

Note: You must have the Current Working Directory *outside* the *src* folder to run the program. The paths for the program are set based on this location.
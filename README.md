# UncertaintyDescent

## Requirement
```
pip install git+https://gitee.com/microcloud/disentanglement_lib.git
```


### Downloading the data sets
To download the data required for training the models, navigate to any folder and run

```
dlib_download_data
```

which will install all the required data files (except for Shapes3D which is not
publicly released) in the current working directory.
For convenience, we recommend to set the environment variable `DISENTANGLEMENT_LIB_DATA` to this path, for example by adding
```
export DISENTANGLEMENT_LIB_DATA=<path to the data directory>
```
to your `.bashrc` file. If you choose not to set the environment variable `DISENTANGLEMENT_LIB_DATA`, disentanglement_lib will always look for the data in your current folder.

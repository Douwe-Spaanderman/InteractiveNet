# InteractiveSeg
Minimally Interactive Segmentation

- [ ] Fingerprinting
- [ ] Preprocessing
- [ ] 
- [ ] 

### Setting environment variables

In order for interactivenet to recognize the locations at which data is stored we have to set paths to these locations. In order to do this we have to export these paths using the following commands:

```
export interactiveseg_raw="~/InteractiveNet/raw"
export interactiveseg_processed="~/InteractiveNet/preprocessed"
export interactiveseg_results="~/InteractiveNet/results/"
```

Note that I have currently set the directories to the current user's home folder `~/`, however you adjust this to your liking.

Using the following commands these locations are stored for the specific session, however can also be set as a global variable by adding these lines somewhere in the `.bashrc` file. This is often located at the user's home folder, hence `~/.bashrc`. This file will most likely be hidden, use `ls -al ~/` to show hidden files in the home directory. If the file is not present use `touch ~/.bashrc` to create it.
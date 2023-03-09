# Using MLflow
All experiments and results are saved at your defined location for interactivenet_results.

Using MLflow is pretty straightforward. First, go to the interactivenet_results folder, where there should a folder named mlruns. Simply run the command:
```
mlflow ui
```

This will create a (gunicorn) server with a user interface you can access. Typically this is hosted on port 5000, so you can access by entering [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser of choise. If port 5000 is in use you can use the ```--port XXXX``` option to specify a different port, e.g. 5020, hence [http://127.0.0.1:5020](http://127.0.0.1:5020).

The mlflow ui has many different options, please refer to [here](https://mlflow.org/) for more indepth information.

## Port forwarding
If you are running your InteractiveNet on a high-performance facility (HPC), your best bet would be to host the mlflow server on the HPC as discussed above, and forward the port, using the following command (Please swap out XXXX for the port, e.g. 5000):
```
ssh -N -L XXXX:localhost:XXXX user@hpc
```
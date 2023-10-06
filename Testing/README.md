# /Testing
Files and code related to the testing and development of the project.

Workflow: Run data_compressor.py, followed by data_augmenter.py to generate samples ready for training the networks on.
Run CNN-LSTM.py to train the CNN-LSTM network, and T.py to train the experimental CNN-LSTM model (with the additional frame selection steps).

When running the readDCA1000 function, make sure to attach a semicolon like so:
readDCA1000('<PATH>');

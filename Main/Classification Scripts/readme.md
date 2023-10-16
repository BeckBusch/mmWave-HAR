# Main\Classification Scripts\
## CNN-LSTM.py
CNN-LSTM implements a 2D CNN-LSTM model, and is used as a baseline model to evaluate the performance of the novel model against
## CNN-LSTM+S.py
CNN-LSTM+S implements a 2D CNN-LSTM with the same network parameters as the baseline, but has additional processing for frame selection and can take input radar sequences of varying lengths
## model_eval.py
model_eval is an evaluation script that is used for testing single activity sequences, it can be used to observe salient points in activity data, and for making predictions using the saved CNN-LSTM+S model
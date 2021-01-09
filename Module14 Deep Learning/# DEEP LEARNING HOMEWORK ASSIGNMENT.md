# DEEP LEARNING HOMEWORK ASSIGNMENT

### Libraries
- Of special note, we import numpy and utilize the random function, setting seed to 1.
- Also, of special note, we import tensor and utilize the random function, setting seed to 2.

### Initial Data Preparation
- We read in two csv files, each into its own Dataframe.   We then merge the relevant data into a single Dataframe, in this case, df.
- We create a function, “window_data” to iterate through df.   The total number of iterations is defined by the total rows in df less the value of “window”.  In the end, we set window = 20.
- While we iterate, we populate two lists, X and y.  Through each iteration we populate X with 20 (window) items from the feature column, while and y with 1 item from the target column.   The thinking is that the current value of y is dependent upon the previous 20 values contained with X.
- Note in the FNG file, X is the fng_value column, while in the Closing file, X is the Close column.   In both cases y is the Close column.
- Finally, we must reshape X & y from horizontal lists to 1 dimensional arrays/vectors each.

### Further Data Preparation
- From the X & y vector, we split each of them within a 70:30 ratio.   The first 70% is for the corresponding training vectors, and the latter 30% is for the testing vectors.
- From sklearn, we use the minmax scaler to normalize the values of X & y.   Each vector’s normalized items will fall within 0 and 1.
- We then reshape again to ensure the model receives a vertical one-dimensional vector.

### Model Preparation
- From tensorflow, we use Sequential, LSTM, Dense, and Dropout.
- Sequential is our model, although we add LTSM at each of the hidden layers.   The number of units (neurons) is 20, the same as our window above.   We have 3 layers, and an output layer.  The output layer is where we use the Dense function.
- The Dropout is set to 0.2 or 20%.   This means as we pass back and forth through the model, 20% of the neurons will be dropped.   These particular neurons have little value added, so dropping them helps the algorithm retain long term memory of earlier epochs.
- The loss ratio is set to mean square error (MSE).   We are trying to minimize this.    We are also using the “Adam” optimizer.

### Running the Model
- I experimented with 3 values for batch_size:  10, 15 and 20.   I did the same for epochs: 10, 15, and 20.  I also varied the window to 10, 15, and 20.   So that’s a total of 3 X 3 X 3 = 27 permutations.   I found 20, 20, 20 to be the best combination.

### Conclusion
- The FNG model does not perform very well.   The MSE remains high for the test data at 0.074, and the chart doesn’t demonstrate a very good fit. 
- The Closing model does better.   The MSE is 0.0138 and the chart shows a reasonable fit.   

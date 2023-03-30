# Analyze-Time-Series-Data-Using-Markov-Transition-Fields
A **Markov transition field (MTF)** is a matrix that indicates the transition probability of the data points at different time steps. A Markov transition field is constructed by unwrapping the Markov Matrix and assigning the transition probabilities to corresponding time steps.

## Required Libraries

* **pandas**: This library will be used for managing the dataset.

* **NumPy**: This library will be used for mathematical functionalities.

* **Scikit-Image**: This library will be used for processing the data matrix.

* **measure**: This module is used to measure different aspects of the data matrix.

* **pyts**: This library is used to process the time series data.

* **preprocessing.discretizer**: This module allows us to discretize a given dataset in bins.

* **Matplotlib**: This library is used for visualization purposes. We will use the following packages:

* **pyplot**: This module will be used for creating inline plots.

* **color**: This module will be used for mapping numbers to RGB colors.

* **cm**: This module will be used for colormap functionalities.

## Dataset
The **Electricity Transformer Temperature (ETT)** dataset comprises the measurements of different parameters of transformers in China. The data was collected from July 2016 to July 2018. For this project, you will use the ETT-small dataset, which contains the data of two transformers. It consists of the following data columns:

|    Column     |           Description            |
| ------------- | -------------------------------- |
| date          | The date of the recorded sample  |
| HUFL          | High Useful Load                 |
| HULL          | High Useless Load                |
| MUFL          | Medium Useful Load               |
| MULL          | Medium Useless Load              |
| LUFL          | Low Useful Load                  |
| LULL          | Low Useless Load                 |
| OT            | Oil Temperature                  |

**Dataset link:**
https://raw.githubusercontent.com/naru94/Analyze-Time-Series-Data-Using-Markov-Transition-Fields/main/dataset/ETTh1.csv

## Steps
1. Import the ETT-small dataset from the following URL into a pandas DataFrame from the repo.
2. Verify the DataFrame by printing a few rows.
3. Find the number of samples in the DataFrame.
4. Truncate the DataFrame to reduce the number of data points.
5. Discretize the HUFL column of the DataFrame into 10 quantile bins using the KBinsDiscretizer() method of the pyts library.
6. Add the discretized HUFL values to a new DataFrame column, HUFL_disc, using the following command: df['label'] = new_data.
7. View the resulting DataFrame.
8. Create an adjacency matrix from the HUFL_disc data column.
9. Print the adjacency matrix.
10. Normalize the columns of the adjacency matrix.
11. Print the resulting matrix.
12. Construct a Markov transition field of the HUFL_disc data.
13. Plot the Markov transition field to visually observe the data pattern in the matrix. 
    * The diagonal entries are generally brighter than the off-diagonal entries, indicating that many samples self-transition instead of transitioning to other bins.
    * The entries adjacent to the diagonal are generally brighter than the entries far away from the diagonal, indicating that there is no abrupt change in the time series data.
    * The probability is highest in the second-last quarter of the Markov transition field (as indicated by the yellow square in the plot), which represents the self-transition of the corresponding data point for a very long time.
14. Downsample the Markov transition field using Piecewise Aggregate Approximation (PAA) to reduce its size while preserving its essential properties.

### Adjacency matrix
An **Adjacency matrix** encodes the dynamics of the time series such that each element indicates the number of times a bin transitions into another bin. The column index represents the previous bin, while the row index represents the next bin.

### Markov matrix
The **Markov matrix** depicts the relation between different datapoints of the time series data. However, it provides little information about the temporal relationship between datapoints. For this purpose, you will need to construct the Markov transition field of the given time series data. A **Markov matrix** indicates the probability of moving from one datapoint to another. So, the only difference between an adjacency matrix and a Markov matrix is that the columns of the latter are normalized, such that the sum of the probabilities of moving to some datapoint from a certain datapoint is always 1.

### Markov transition field (MTF)
A **Markov transition field (MTF)** is a matrix that indicates the transition probability of the datapoints at different time steps. A Markov transition field is constructed by unwrapping the Markov Matrix and assigning the transition probabilities to corresponding time steps.

### Self-transition probability
Self-transition probability is an important parameter of the time series data, which represents the probability of a datapoint to transition to itself after a time step. In a Markov transition field, this is depicted by the diagonal entries of the matrix.

### Piecewise Aggregate Approximation (PAA)
**Piecewise Aggregate Approximation (PAA)** is a data-reduction technique in which an n-sized data is divided into w blocks, and each block contains the mean of the comprising samples.

## Code
The code is plotting a line graph of the 'HUFL_disc' column from a Pandas DataFrame called 'ett' against time, with the color of each line segment determined by the corresponding value in the 'mtf_diag' list.

Here's a breakdown of the code:

mtf_diag = [mtf_reduced[i][i] for i in range(len(mtf_reduced))] - This creates a new list called 'mtf_diag' by extracting the diagonal values from a matrix called 'mtf_reduced'.

fig, ax = plt.subplots(figsize = (28, 4)) - This creates a new figure and axes object for the plot.

norm = colors.Normalize(vmin=np.min(mtf_diag), vmax=np.max(mtf_diag)) - This creates a normalization object for the color mapping, with the minimum and maximum values of 'mtf_diag' as the lower and upper bounds.

cmap = cm.viridis - This sets the color map to 'viridis'.

for i in range(0, n_t, 10): - This loops through the range of integers from 0 to 'n_t' with a step of 10.

ax.plot(ett['date'][i:i+10+1], ett['HUFL_disc'][i:i+10+1], c = cmap(norm(mtf_diag[int(i/10)]))) - This plots a line graph of the 'HUFL_disc' column from the 'ett' DataFrame against the corresponding values in the 'date' column, for the subset of rows from i to i+10+1. The color of the line segment is determined by the corresponding value in the 'mtf_diag' list, normalized by the normalization object created earlier.

plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax) - This creates a color bar for the plot, with the same normalization and color map objects.

plt.show() - This displays the plot.

import matplotlib.pyplot as plt

# Define the list of N_Part_Values
N_Part_Values = [10, 20, 30, 40, 50]

# Loop over the N_Part_Values and plot the data for each iteration
for i, n in enumerate(N_Part_Values):
    # Plot the 'Pt' column where N_part == n
    plt.scatter(mergedData['Pt'][mergedData['N part'] == n], 
                mergedData['Spectrum'][mergedData['N part'] == n], 
                color='C{}'.format(i), 
                label='N_part = {}'.format(n))

    # Plot the 'predictions' column where N_part == n
    plt.scatter(mergedData['Pt'][mergedData['N part'] == n], 
                mergedData['predictions'][mergedData['N part'] == n], 
                color='C{}'.format(i), 
                label='_nolegend_')

# Add a legend and axis labels
plt.legend()
plt.xlabel('Pt')
plt.ylabel('Value')

# Show the plot
plt.show()
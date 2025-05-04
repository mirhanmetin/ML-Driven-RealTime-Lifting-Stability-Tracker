import matplotlib.pyplot as plt

# Visualizes the results of the anomaly detection process.
# It creates a scatter plot of the left and right foot pressure data, coloring the points based on their anomaly status.
# The function takes a DataFrame containing the results of the anomaly detection process as input.

# @param results: DataFrame containing the results of the anomaly detection process.
# @return: None
# @plot_anomalies = Function to visualize the results of the anomaly detection process.
def plot_anomalies(results):
    plt.ion() # Enables interactive mode for plotting.
    fig, ax = plt.subplots(figsize=(12, 7)) # Creates a figure and a set of subplots
    ax.set_title("Anomaly Detection", fontsize=14) # Sets the title of the plot
    ax.set_xlabel("Left Foot Pressure") # Sets the x-axis label of the plot
    ax.set_ylabel("Right Foot Pressure") # Sets the y-axis label of the plot
    ax.grid(True) # Enables the grid on the plot

    for i in range(len(results)):
        row = results.iloc[i]
        x = row['left_foot_pressure']
        y = row['right_foot_pressure']

        if row['Final_Anomaly']:
            color = 'red'
        elif "UYARI!" in row['Logic_Alert']:
            color = 'orange'
        else:
            color = 'green'

        ax.scatter(x, y, color=color, edgecolors='k', s=80) # Creates a scatter plot of the left and right foot pressure data
        ax.annotate(str(i + 1), (x + 0.005, y + 0.005), fontsize=7) # Annotates the points in the scatter plot with their index
        plt.pause(0.05) # Pauses the plot for a short period to allow for dynamic updates

    plt.ioff() # Disables interactive mode for plotting
    plt.show() # Displays the plot

from imports import pd, plt, sns

def plot_univarate(data, column, title=None, x_axis_t=None, figsize=(8,4), color='skyblue', bins=20, kde=False):
    """
    Plot the distribution of a single variable.
    
    Args:
        data (pd.DataFrame): The dataset.
        column (str): The column to analyze.
        title (str): The title of the plot.
        x_axis_t (str): Label for the x-axis.
        color (str): The color of the bars. Default is 'skyblue'.
        bins (int): Number of bins for the histogram. Default is 20.
        kde (bool): Whether to plot a KDE curve. Default is False.
    """
    plt.figure(figsize=figsize)
    sns.histplot(data[column], bins=bins, kde=kde, color=color, edgecolor='black')
    plt.title(title or f"Univariate Analysis of {column}")
    plt.xlabel(x_axis_t or column)
    plt.ylabel('Frequency')
    plt.show()

def plot_bivariate(data, x_column, y_column, plot_type='scatter', title=None, x_axis_t=None, y_axis_t=None, color='skyblue'):
    """
    Plot the relationship between two variables.
    
    Args:
        data (pd.DataFrame): The dataset.
        x_column (str): The column for the x-axis.
        y_column (str): The column for the y-axis.
        plot_type (str): Type of plot ('scatter', 'line', 'heatmap'). Default is 'scatter'.
        title (str): The title of the plot.
        x_axis_t (str): Label for the x-axis.
        y_axis_t (str): Label for the y-axis.
        color (str): Color of the plot elements. Default is 'skyblue'.
    """
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'scatter':
        sns.scatterplot(data=data, x=x_column, y=y_column, color=color)
    elif plot_type == 'line':
        sns.lineplot(data=data, x=x_column, y=y_column, color=color, marker='o')
    elif plot_type == 'heatmap':
        pivot_table = data.pivot_table(values=y_column, index=x_column, aggfunc='mean').fillna(0)
        sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.1f')
    else:
        raise ValueError("Invalid plot_type. Use 'scatter', 'line', or 'heatmap'.")
    
    plt.title(title or f"Bivariate Analysis: {x_column} vs {y_column}")
    plt.xlabel(x_axis_t or x_column)
    plt.ylabel(y_axis_t or y_column)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_trends(data, title, x_label, y_label, plot_type='line', figsize=(12, 6), color='blue', xticks_rotation=0, reduce_xticks=False, xticks_step=1):
    """
    Plots trends over time for a given dataset.

    Args:
        data (pd.Series or pd.DataFrame): Data to plot, typically an aggregated Series (e.g., daily, monthly counts).
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        plot_type (str): Type of plot ('line', 'bar'). Defaults to 'line'.
        figsize (tuple): Size of the figure. Defaults to (12, 6).
        color (str): Color of the plot. Defaults to 'blue'.
        xticks_rotation (int): Rotation angle for x-axis tick labels. Defaults to 0.
        reduce_xticks (bool): Whether to reduce the number of x-axis labels. Defaults to False.
        xticks_step (int): Step size for reducing x-axis labels. Defaults to 1.
    """
    plt.figure(figsize=figsize)
    
    # Plot the data
    if plot_type == 'line':
        data.plot(kind='line', color=color, title=title)
    elif plot_type == 'bar':
        ax = data.plot(kind='bar', color=color, title=title)
        if reduce_xticks:
            ticks_to_display = range(0, len(data), xticks_step)
            ax.set_xticks(ticks_to_display)
            ax.set_xticklabels(data.index[ticks_to_display], rotation=xticks_rotation)
    else:
        raise ValueError("Invalid plot_type. Use 'line' or 'bar'.")
    
    # Set labels and grid
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y')
    
    # Rotate x-axis ticks if needed
    if not reduce_xticks:
        plt.xticks(rotation=xticks_rotation)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

import numpy as np
import re
from matplotlib import pyplot as plt
import lime
import lime.lime_tabular as lt

# LIME processing functions

def spectra_explainer(data, spectra_length):
    # Initialize a LIME explainer object
    
    # Note: You might need to adjust the feature names and class names according to your dataset
    explainer = lt.LimeTabularExplainer(training_data=np.array(data), # Use your data here
                                                    mode='classification', # or 'classification' based on your model
                                                    feature_names = [f'*{i}*' for i in range(spectra_length)],
                                                    discretize_continuous=True, discretizer='decile')
    return explainer


def model_predict(model, data):
    return model.predict(data)

def calculate_lime(categories, explainer, x_axis_values, predict_fn, scale_factor):

    """
    Mass calculate LIME explanations for all instances across all categorical outcomes in the analysis.

    Parameters:
    - explainer: lime explainer
    - categories: list of numpy arrays containing all spectra within each category. 
    A category can correspond to bucketed regression outcomes or to categorical prediction outcomes.
    - x_axis_values: x axis values of the spectra
    - predict_fn: function taking a 2D numpy array of perturbed samples and returning model predictions

    Returns:
    - lime_data: list of lime explanations for all instances
    - category_indicator: list indicating which category a given explanation is for
    - spectra_indicator: list indicating where the spectra is in the original dataset
    - mean_spectra_list: list of mean spectra across all categories

    """


    lime_data = []
    category_indicator = []
    spectra_indicator = []
    mean_spectra_list = []
    spectra_length = len(x_axis_values)

    for category_i in range(len(categories)):
        for spectra_i in range(len(categories[category_i])):
            instance_to_explain = categories[category_i][spectra_i] # Adjust index based on which instance you want to explain
            
            exp = explainer.explain_instance(data_row=instance_to_explain, 
                                predict_fn=predict_fn,
                                num_features=spectra_length,
                                num_samples=5000
                                )

            # Get the list of explanations for all the features
            weights = exp.as_list()
            # print(weights)
            weights = np.array(sort_lime(weights, instance_to_explain, x_axis_values, scale_factor=scale_factor))
            lime_data.append(weights)
            category_indicator.append(category_i)
            spectra_indicator.append(spectra_i)
            mean_spectra_list.append(instance_to_explain)
    return lime_data, category_indicator, spectra_indicator, mean_spectra_list


def sort_lime(lime_weights, mean_spectra, x_axis_values, scale_factor =1):

    """
    The LIME functions typically output the feature weights in order of size instead of in order of appearance.
    The present function organizes the weights for ease of plotting. 

    Parameters:
    - lime_weights: lime weights
    - mean_spectra: mean spectra for set lime weights 
    - x_axis_values: x axis values of the spectra

    Returns:
    - extracted_data: returns a list of lime weights in order with x axis values combined

    """
    # print(lime_weights)
    # print(len(lime_weights))
    # Regex pattern to capture comparison operators and values
    # pattern = r"(?:(\d+\.\d+) < )?\*(\d+)\*\s*(<=?|>=?)\s*(\d+\.\d+)(?:\s*(<=?|>=?)\s*(\d+\.\d+))?"
    pattern = r"(?:(-?\d+\.\d+) < )?\*(\d+)\*\s*(<=?|>=?)\s*(-?\d+\.\d+)(?:\s*(<=?|>=?)\s*(-?\d+\.\d+))?"
    extracted_data = []

    for feature, weight in lime_weights:
        # Find the index
        match = re.search(pattern, feature)
        # print(feature)
        # print(match)
        if match:
            index = int(match.group(2))
            direction = match.group(3)
            value1 = (match.group(1))
            value2 = (match.group(4))
            # lower_bound = 0
            # upper_bound = 1
            if value1 is not None:
                lower_bound = float(value1)
                upper_bound = float(value2)
            elif direction == '>' or '>=':
                upper_bound = float(value2)
                lower_bound = mean_spectra[index]
            else:
                lower_bound = float(value2)
                upper_bound = mean_spectra[index]
        else:
            print(f"Regex did not match for feature: {feature}")
            continue

        # Store extracted and processed data
        extracted_data.append((index, lower_bound/scale_factor, upper_bound/scale_factor, weight))

    # Sorting by the feature index
    extracted_data.sort(key=lambda x: x[0])
    extracted_data = np.array(extracted_data)
    # Map feature indices to actual x-axis values
    xa = np.array(x_axis_values).flatten()
    # Replace the index column with corresponding x-axis values
    extracted_data[:, 0] = xa[extracted_data[:, 0].astype(int)]
    return extracted_data


# Function assumes that spectra are categorized for example according to concentration ranges
# If only one category exists, input the data as a list of one category
# def combine_lime(explainer, categories, range_of_spectra):

#   category_arrays = []
#   num_categories = len(categories)
#   for i in range(len(categories)):
#     # Create lists to hold the data for each column across all arrays
#     column_data = [[] for _ in range(num_categories)]
#     mean_spectra = np.mean(categories[i][0:range_of_spectra], axis=0)
#     spectra_length = len(mean_spectra)
#     for j in range(len(categories[i])):
#       # Select an instance to explain
#       instance_to_explain = categories[i][j]

#       # Generate LIME explanation for this instance
#       exp = explainer.explain_instance(data_row=instance_to_explain, 
#                                       predict_fn=model_predict,
#                                       num_features=spectra_length)
#       # Get the list of explanations for all the features
#       weights = exp.as_list()
#       weights = np.array(sort_lime(weights, mean_spectra))
#       # Append each column of each array to the respective list
#       for col in range(num_categories):
#         column_data[col].append(weights[:,col])
      
#       # Convert lists to arrays and compute the mean across rows (axis=0)
#     mean_columns = [np.mean(np.stack(cols), axis=0) for cols in column_data]
#       # Combine the mean columns back into a single array
#     final_mean_array = np.column_stack(mean_columns)
#     category_arrays.append(final_mean_array)
#   return category_arrays


def plot_lime_global(plot_data, mean_spectra, title, cluster_indices = None, plot_mean_spectra = True, plot_shade = True, plot_clusters = False, figsize=(19, 5), lime_scale_factor = 10):

    """
    LIME explanation plotting function.
    The function is also used to plot lime explanations for crime contexts which represent the mean LIME explanation in set context.
    Toggle options extst for plotting perturbation limits (shade), highlight clusters for crime (clusters) or mean spectra for clarity (plot_mean_spectra).

    Parameters:
    - plot_data: input LIME data from earlier function
    - mean_spectra: array of mean spectra for set lime weights 
    - title: figure title
    - cluster_indices: list of highlight cluster indices extracted from corresponding crime functions (top_cluster_indices_global)
    - plot_mean_spectra: boolean toggle option for plotting mean spectra for set instance or crime context
    - plot_shade: boolean toggle option for plotting LIME perturbation limits. 
      Shades can be sometimes misleading if in a particular crime context the upper limit is near limits but not always leading to a narrower area then in reality.
    - plot_clusters: boolean toggle option for plotting highlighted areas. Areas of interest will remain visible while areas removed in later stages are filled in with black.
    - figsize: figure sizing option

    Returns:
    - fig: a matplotlib figure of the LIME/crime explanation

    """


    fig = plt.figure(figsize=figsize)

    # Scale up LIME data for better visibility
    plot_data[:,3] = plot_data[:,3] * lime_scale_factor
    
    if plot_mean_spectra:
      # Start plotting the mean spectra, using default black color initially
      last_index = 0
      current_color = 'black'
      plt.plot(plot_data[:, 0], mean_spectra, c = current_color, label = "Mean Context Spectrum")
      for i in range(1, len(plot_data[:, 0])):
          # Check if current point matches the mean spectra
          if (plot_data[i, 1] == mean_spectra[i]) or (plot_data[i, 2] == mean_spectra[i]):
              color = 'red'
          else:
              color = 'black'
          
          # If color changes, plot the previous segment
          if color != current_color:
              plt.plot(plot_data[last_index:i, 0], mean_spectra[last_index:i], c=current_color, linestyle = 'solid', linewidth = 1)
              last_index = i
              current_color = color
      
      # Plot the last segment
      plt.plot(plot_data[last_index:, 0], mean_spectra[last_index:], c=current_color)

    if(plot_shade):
        # Shade the area between plot_data[:,1] and plot_data[:,2]
        plt.fill_between(plot_data[:, 0], plot_data[:,1], plot_data[:,2], color='teal', alpha=0.5)
        # Separating positive and negative values for plot_data[:,3]
        positives = plot_data[:,3] >= 0
        negatives = plot_data[:,3] < 0

        # Filling positive and negative values with different colors
        plt.fill_between(plot_data[:, 0], 0, plot_data[:,3], where=positives, color='darkgreen', alpha=0.9, label=f'{lime_scale_factor}x Positive LIME Weights')
        plt.fill_between(plot_data[:, 0], 0, np.abs(plot_data[:,3]), where=negatives, color='orange', alpha=0.9, label=f'{lime_scale_factor}x Negative LIME Weights')
    else:
        # Separating positive and negative values for plot_data[:,3]
        positives = plot_data[:,3] >= 0
        negatives = plot_data[:,3] < 0

        # Filling positive and negative values with different colors
        plt.fill_between(plot_data[:, 0], 0, plot_data[:,3], where=positives, color='darkgreen', alpha=0.9, label=f'{lime_scale_factor} x Positive LIME Weights')
        plt.fill_between(plot_data[:, 0], 0, np.abs(plot_data[:,3]), where=negatives, color='orange', alpha=0.9, label=f'{lime_scale_factor} x Negative LIME Weights')

    if(plot_clusters):
        # Define colors for top clusters - adjust colors as needed

        # Shade the areas corresponding to cluster indices
        for idx, indices in enumerate(cluster_indices):
            # color = colors[idx % len(colors)]
            for index in indices:
                plt.fill_betweenx([0, mean_spectra[index]], plot_data[index, 0] - 0.5, plot_data[index, 0] + 0.5, color='black', alpha=0.9)


    # plt.ylim(0, 1)
    marker_size = 17
    label_size = 19
    plt.xticks(fontsize=marker_size)
    plt.yticks(fontsize=marker_size)
    plt.tick_params(axis='x', direction='in', width=4, labelsize=marker_size, colors='black')
    plt.tick_params(axis='y', direction='in', width=4, labelsize=marker_size, colors='black')
    plt.grid(False)
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    # plt.legend(markerscale = 3, fontsize=label_size)
    plt.title(title, color='black', fontsize=label_size)
    plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=label_size)
    plt.ylabel('Relative Intensity', fontsize=label_size)
    # axs.tick_params(axis='x', direction='in', width=4, labelsize=16, colors='black')
    # axs.tick_params(axis='y', direction='in', width=4, labelsize=16, colors='black')
    plt.tight_layout()
    # plt.show()
    # fig.savefig(f'crime{title}.pdf', dpi = 600)
    return(fig)


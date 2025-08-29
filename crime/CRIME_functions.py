from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from crime.lime_processing_functions import plot_lime_global
from sklearn.preprocessing import MinMaxScaler
from matplotlib import colormaps
from crime.CRIME_utils import cosine_similarity_manual
from matplotlib.patches import Rectangle
import torch
import umap
from torch.utils.data import DataLoader, TensorDataset

# crime Functions


def CRIME_fit(number_of_clusters, latent_space, weight_data, mean_spectra_list, random_state = 42):

    """
    Identifies crime clusters using K-means clustering on the LIME explanations.

    Parameters:
    - number_of_clusters: identified number of contexts from visualising the crime space
    - latent_space: produced latent space by crime autoencoder of choice
    - weight_data: LIME explanation weights
    - mean_spectra_list: List of mean spectra within each category

    Returns:
    - separated_arrays: dict of LIME weight data for each spectra within set context
    - separated_spectra: dict of spectra from each context
    - spectra_means: dict of mean spectra for each context
    - context_labels: context labels for each spectra in the original dataset
    """
    
    # KMeans clustering
    
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state)
    kmeans.fit(latent_space)
    context_labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    # Finding unique labels
    unique_labels = np.unique(context_labels)

    # Creating dictionaries to store arrays by label
    separated_arrays = {}
    separated_spectra = {}
    spectra_means = {}

    for label in unique_labels:
        indices = np.where(context_labels == label)[0]  # Get indices where labels match the current label
        separated_arrays[label] = weight_data[indices]  # Index the weight data array
        separated_spectra[label] = np.array(mean_spectra_list)[indices]  # Index the spectra data array

        # Calculate mean spectra for each label
        spectra_means[label] = np.mean(separated_spectra[label], axis=0)
    return separated_arrays, separated_spectra, spectra_means, context_labels

def plot_CRIME(names, context_names, crime_labels, latent_space, category_indicator):
    
    """
    Plots the crime contexts from the latent space as well as the categorical groupings.

    Parameters:
    - names: list of names of categories plotted
    - context_names: list of names of contexts
    - crime_labels: list of labels for each crime spectra
    - latent_space: crime latent space
    - category_indicator: list of labels for ground truth categories

    Returns: Figure

    """

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_space)
    # Set default font size
    # rcParams['font.size'] = 14
    fig1, ax1 = plt.subplots(figsize=(18, 6), nrows = 1, ncols = 2)
    fig1.subplots_adjust(right=0.8, wspace=0.35)
    # Scatter plot of the clusters
    # print(np.shape(embedding))
    ax1[0].scatter(embedding[:, 0], embedding[:, 1], c=crime_labels, cmap='viridis', edgecolors='grey')

    # Creating a custom legend for clusters
    colors = plt.cm.viridis(np.linspace(0, 1, len(context_names)))  # get the colors of the current colormap
    for i, color in enumerate(colors):
        ax1[0].scatter([], [], color=color, label=f'Context {context_names[i]}')

    # plt.colorbar()
    ax1[0].set_xlabel('UMAP Dimension 1', color='black')
    ax1[0].set_ylabel('UMAP Dimension 2', color='black')
    # ax1[0].set_title('UMAP Latent Space Representation Colored by Context', color='black')
    # ax1[0].legend(loc = 'right', markerscale = 2)
    ax1[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., markerscale=2)

    # Unique categories and colors
    unique_categories = np.unique(category_indicator)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_categories)))  # generate colors

    for i, cat in enumerate(unique_categories):
        inds = [j for j, x in enumerate(category_indicator) if x == cat]
        ax1[1].scatter(embedding[inds, 0], embedding[inds, 1], color=colors[i], label=f'{names[i]}', alpha=0.8)

    # plt.colorbar()
    ax1[1].set_xlabel('UMAP Dimension 1', color='black')
    ax1[0].set_xlim(-5,15.5)
    ax1[1].set_xlim(-5,15.5)
    ax1[1].set_ylim(-3,14)
    ax1[0].set_ylim(-3,14)
    # ax1[1].set_ylabel('Latent Dimension 2')
    # ax1[1].set_title('UMAP Latent Space Representation Colored by Category', color='black')
    ax1[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., markerscale=2)
    plt.show()
    return fig1

def CRIME_clustering(separated_arrays, spectra_means, context_names, plot_clusters = False):

    """
    This function both plots the separate crime contexts according to the mean spectra and weights of each.
    Separately it clusters the context spectra according to position, height, and LIME weight of each spectra.
    From this clustering, the top 5 clusters are selected to represent the key areas of each spectra corresponding to a compound.


    Parameters:
    - separated_arrays: list of LIME weight data
    - spectra_means: mean spectra for each context
    - context_names: list of names of contexts
    - plot_clusters: boolean toggle option to visualise the clustering effect

    Returns: 
    - figs: list of figures produced for each context
    - second_figs: list of figures produced for each context following clustering
    - top_cluster_indices_global: list of indices for cluster regions which should be highlighted in the crime matching step

    """
    top_cluster_indices_global = []
    
    figs = []
    second_figs = []

    for i in range(len(separated_arrays)):
        # Step 1: Stack the arrays along a new axis, creating a 3D array
        stacked_arrays = np.stack(separated_arrays[i])
        
        # Step 2: Compute the mean along the new axis (axis=0), which will reduce the 3D array back to a 2D array
        mean_of_positions = np.mean(stacked_arrays, axis=0)
        mean_spectra = spectra_means[i]
        
        # Example data
        spectra = mean_spectra
        weights = mean_of_positions[:,3]
        positions = mean_of_positions[:,0]

        # Reshape data for scaling if necessary
        spectra = np.array(spectra).reshape(-1, 1)
        weights = np.array(weights).reshape(-1, 1)
        positions = np.array(positions).reshape(-1, 1)

        figs.append(plot_lime_global(mean_of_positions,  mean_spectra, f'CRIME Context {context_names[i]}', lime_scale_factor=20))

    

    for j in range(len(separated_arrays)):

        # Initialize scaler with the desired range
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # Scale both arrays
        spectra_scaled = scaler.fit_transform(spectra)
        weights_scaled = scaler.fit_transform(weights)
        position_scaled = scaler.fit_transform(positions)
        # # Combine x, y, z into a single 2D array
        
        # Combine x, y, z into a single 2D array
        X = np.column_stack((position_scaled, weights_scaled, spectra_scaled))  # Transpose to make sure each row is (x, y, z)
        # print(spectra)
        # print(weights)
        # print(positions)
        # print(X)
        
        n_clusters=15
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters, n_init=15)
        kmeans.fit(X)
        scatterweight_labels = kmeans.labels_

        # Calculate the prominence for each cluster
        cluster_prominence = []
        for i in range(n_clusters):
            cluster_points = X[scatterweight_labels == i]
            # Assuming weight is in the second column and spectra in the second column
            total_weight_spectra = np.sum(np.abs(cluster_points[:, 1]) * (cluster_points[:,2]))
            cluster_prominence.append((total_weight_spectra, i))

        # Sort clusters by the total weight and spectra sum (prominence)
        sorted_clusters = sorted(cluster_prominence, key=lambda x: x[0], reverse=True)  # Higher sum first

        # Select top 5 prominent clusters
        top_5_clusters = [x[1] for x in sorted_clusters[:5]]
        bottom_5_clusters = [x[1] for x in sorted_clusters[5:]]

        # This will give you a list of arrays, each containing the indices of points in one of the top 5 clusters
        # top_cluster_indices = ([np.where(scatterweight_labels == cluster_id)[0] for cluster_id in top_5_clusters])
        bottom_cluster_indices = ([np.where(scatterweight_labels == cluster_id)[0] for cluster_id in bottom_5_clusters])

        top_cluster_indices_global.append(bottom_cluster_indices)

        if plot_clusters:

            # Creating a figure
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            # Get a colormap with distinct colors for the top 5 clusters
            cmap = colormaps.get_cmap('Set1')

            # Plotting only the top 5 clusters using the selected colormap
            for cluster_id in top_5_clusters:
                cluster_points = X[scatterweight_labels == cluster_id]
                ax.scatter(*cluster_points.T, color=cmap(top_5_clusters.index(cluster_id)), edgecolors = 'black')  # Map color index correctly

            ax.set_xlabel('Position Z-score')
            ax.set_ylabel('Weights scaled')
            ax.set_zlabel('Spectra intensity scaled')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.show()

            # mean_of_positions is a 2D array (842x4) where each element is the mean of that position across all arrays
            second_figs.append(plot_lime_global(mean_of_positions,  mean_spectra, f'crime Context {context_names[j]}', top_cluster_indices, True, True, True, lime_scale_factor=1))
    return figs, second_figs, top_cluster_indices_global

def run_CRIME(lime_data, encoder, cat_names, context_names, mean_spectra_list, category_indicator, plot_clusters = False, random_state = 42, lime_weights_only=False):
    
    """
    The main crime function which runs all other functions with the exception of the similarity match.
    Produces two latent space figures colored by both category and context, as well as plots for the mean explanation for all contexts
    as well as optionally the selected regions of each mean context explanation for further assessment.
    Output of the function can be used separately or for the similarity match function.

    Parameters:
    - lime_data: list of LIME explanations output from calculating all lime explanations
    - encoder: trained crime autoencoder of choice
    - cat_names: list of names for each outcome category
    - context_names: list of names for each context
    - mean_spectra_list: list of mean spectra from each category
    - category_indicator: list of labels indicating the category a spectra belongs to in the original dataset
    - plot_clusters: boolean toggle indicator for desire to plot black bars around regions of non-interest
    - random_state: random seed

    Returns: 
    - separated_arrays: dict of LIME weight data for each spectra within set context
    - separated_spectra: dict of spectra from each context
    - spectra_means: dict of mean spectra for each context
    - crime_labels: list of context labels for each spectra in the original dataset
    - figs: list of figures produced for each context
    - second_figs: list of figures produced for each context following clustering
    - top_cluster_indices_global: list of indices for cluster regions which should be highlighted in the crime matching step

    """
    # Number of contexts is derived from length of naming array
    number_of_clusters = len(context_names)
    # We use only the last three columns for the VAE
    if lime_weights_only:
        latent_space_data = lime_data[:,:, 3:]
    else:
        latent_space_data = lime_data[:, :, 1:]
    # print(latent_space_data[0])
    weight_data = lime_data
    print("C")
    
    print(latent_space_data.shape)
    reshaped_latent_space_data = torch.tensor(latent_space_data.reshape(latent_space_data.shape[0], -1))
    print(reshaped_latent_space_data.shape)
    print(torch.is_tensor(reshaped_latent_space_data))
    # Predict latent space
    # Make a loader for this to then pass to the encoder
    loader = DataLoader(TensorDataset(reshaped_latent_space_data), batch_size=20, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Made loader")
    mu = []
    logvar = []
    encoder.to(device)
    for batch in loader:
        # print(batch)
        batch_rlsd = batch[0]
        batch_rlsd = batch_rlsd.to(device)
        _, _, batch_mu, batch_logvar = encoder(batch_rlsd)
        # print("bm", batch_mu)
        # print("bl", batch_logvar)
        mu = torch.cat((torch.tensor(mu).to(device), torch.tensor(batch_mu).to(device)), dim=0)
        logvar = torch.concat((torch.tensor(logvar).to(device), torch.tensor(batch_logvar).to(device)), dim=0)
        # print(mu)
        # print(logvar)
        # print(len(mu))
        # print(len(logvar))
    # mu, logvar = encoder.predict(reshaped_latent_space_data)
    # print("Did predictions")
    latent_space = torch.cat((torch.tensor(mu), torch.tensor(logvar)), dim=1).detach().cpu().numpy()
    # print((latent_space[0][0]))
    separated_arrays, separated_spectra, spectra_means, crime_labels = CRIME_fit(number_of_clusters, latent_space, weight_data, mean_spectra_list, random_state)

    crime_figs = plot_CRIME(cat_names, context_names, crime_labels, latent_space, category_indicator)
    
    figs, second_figs, top_cluster_indices_global = CRIME_clustering(separated_arrays, spectra_means, context_names, plot_clusters)

    return separated_arrays, separated_spectra, spectra_means, crime_labels, figs, second_figs, top_cluster_indices_global, crime_figs
    

def similarity_match(target_spectra, target_titles, target_colors, separated_arrays, top_cluster_indices_global, spectra_means, show_plots=False):
    
    """
    Final crime function to match the identified crime contexts with target compound spectra.
    Matching is done using cosine similarity, and the function additionally plots sanity check visualisations
    of the weighted outcomes. The cosine similarity is applied to spectra weighted according to the mean context weights
    and an identical weighing is applied on the baseline target spectra as well to enhance similarities in key areas.
    Similarly, the spectra are both cut to only contain the highlighted regions of interest identified in earlier functions.
    The final output is a list providing the index of the highest matching target compound as well as all similarity scores.
    Similarity scores range from -1 to 1.

    Parameters:
    - target_spectra: list of LIME explanations output from calculating all lime explanations
    - target_titles: trained crime autoencoder of choice
    - target_colors: list of names for each outcome category
    - separated_arrays: dict of LIME weight data for each spectra within set context
    - top_cluster_indices_global: list of indices for cluster regions which should be highlighted
    - spectra_means: dict of mean spectra for each context


    Returns: 
    - max_similarity_target: list of indexes representing the highest scoring target
    - combined_similarities: list of cosine similarity values for each target in each context

    """
    
    
    # Example list of arrays with some arrays possibly containing nan values
    list_of_arrays = top_cluster_indices_global
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Initialize an empty list to store all valid indices
    final_indices = []

    # Loop through each array and collect non-nan values
    for arr in list_of_arrays:
        all_indices = []
        for i in arr:
            # Filter out nan values using boolean indexing
            filtered_array = i[~np.isnan(i)]
            all_indices.extend(filtered_array)
        
        final_indices.append(all_indices)


    print("A")
    max_similarity_target = []
    combined_similarities = []

    for j in range(len(separated_arrays)):
        # Step 1: Stack the arrays along a new axis, creating a 3D array
        stacked_arrays = np.stack(separated_arrays[j])
        # Gives an array of shape (number of spectra in a context, 801, 4)

        # Step 2: Compute the mean along the new axis (axis=0), which will reduce the 3D array back to a 2D array
        mean_of_positions = np.mean(stacked_arrays, axis=0)
        # This gives the average values for the lime weights of the spectra within each context
        # print(np.shape(mean_of_positions))
        mean_spectra = spectra_means[j] # The mean spectra associated with the context

        weights = mean_of_positions[:,3]

        weights = np.array(weights).reshape(-1, 1)
        # print(np.shape(weights)) 

        # Initialize scaler with the desired range
        scaler = MinMaxScaler(feature_range=(-1, 1))

        weights_scaled = scaler.fit_transform(weights)
        # Example data
        spectra = mean_spectra
        # Reshape data for scaling if necessary
        spectra = np.array(spectra).reshape(-1, 1)
        # Scale both arrays
        spectra_scaled = scaler.fit_transform(spectra)

        spectra_scaled = spectra_scaled*weights_scaled

        spectra_unweighed = scaler.fit_transform(spectra)

        indices_to_remove = np.unique(final_indices[j])
        # x_axis_cut = np.delete(x_axis_values, indices_to_remove)
        spectra_scaled = np.delete(spectra_scaled, indices_to_remove).reshape(-1, 1)
        spectra_unweighed = np.delete(spectra_unweighed, indices_to_remove).reshape(-1, 1)
        # Cut weights to match removed indices for target weighting
        weights_scaled = np.delete(weights_scaled, indices_to_remove, axis=0).reshape(-1, 1)

        if show_plots:
            fig, ax = plt.subplots(figsize=(6, 2*len(target_spectra)), nrows=len(target_spectra), ncols=1)
        # Vectorized scaling and removal of unwanted indices for all target spectra at once
        # Stack into shape (n_targets, spectrum_length)
        targets_arr = np.stack([ts.reshape(-1) for ts in target_spectra])
        # Transpose to (spectrum_length, n_targets) so each column is a spectrum
        targets_arr_T = targets_arr.T
        # Scale each column (spectrum) between -1 and 1 in one go
        scaled_targets = MinMaxScaler(feature_range=(-1, 1)).fit_transform(targets_arr_T)
        # Remove rows corresponding to indices_to_remove
        cut_scaled_targets = np.delete(scaled_targets, indices_to_remove, axis=0)
        # Split back into list of (length, 1) arrays
        combined_cut_target = [
            cut_scaled_targets[:, i].reshape(-1, 1)
            for i in range(cut_scaled_targets.shape[1])
        ]
            
        print(j)
        # Vectorized similarity computation
        # Use the precomputed cut_scaled_targets (shape: n_pts × n_targets)
        Y = cut_scaled_targets * weights_scaled  # broadcasts weights over columns
        x = spectra_scaled.flatten()
        # Compute dot products and norms
        dot_prods = x @ Y
        norm_x = np.linalg.norm(x)
        norms_y = np.linalg.norm(Y, axis=0)
        # Cosine similarities for all targets at once
        similarities = (dot_prods / (norm_x * norms_y)).tolist()
        combined_similarities.append(similarities)
        max_similarity_target.append(int(np.argmax(similarities)))

        # Plotting if requested
        if show_plots:
            for i, sim in enumerate(similarities):
                ax[i].plot(x, color='black')
                ax[i].plot(Y[:, i])
                ax[i].set_ylim(-1, 1)
                ax[i].set_xlabel(f'{target_titles[i]}: {sim: .3f}')
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                for spine in ax[i].spines.values():
                    spine.set_visible(False)
        
        if show_plots:
        # Collect all handles and labels
            handles, labels = [], []
            for axis in ax.flat:
                for handle, label in zip(*axis.get_legend_handles_labels()):
                    handles.append(handle)
                    labels.append(label)

            # Create a single legend for the whole figure with all handles and labels
            fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
            ax[1].set_ylabel('Cluster region LIME weighted spectra')
            ax[0].set_title(f'Cosine Similarity for Cluster {j+1}', weight = 'bold')

            rect = Rectangle((0, 0), 1, 1, fill=False, color="white", linewidth=2, transform=fig.transFigure, clip_on=False)
            fig.patches.append(rect)

            plt.show()
    return max_similarity_target, combined_similarities

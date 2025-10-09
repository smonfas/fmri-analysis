#!/usr/bin/env python3
"""
Functions for plotting and visualization on (flat) surface meshes.
"""

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

def load_surf_data(fname):
    """ Load surface data from a file, either a CIFTI or GIFTI (metric) file."""
    if fname.endswith('.dscalar.nii'):
        data = nib.load(fname).get_fdata()[0,:163842]
    elif fname.endswith('.func.gii') or fname.endswith('.shape.gii'):
        data = nib.load(fname).agg_data()
    else:
        raise ValueError(f'Unknown file type: {fname}')
    return data

def load_surface_gifti(fname):
    surf_gii = nib.load(fname)
    vertices = surf_gii.darrays[0].data
    faces = surf_gii.darrays[1].data    
    surf = dict(vertices=vertices, faces=faces)
    return surf

def find_boundary_edges(surf, roi_mask=None):
    """ Find the boundary edges of an ROI in a surface mesh."""
    faces = surf['faces']
    vertices = surf['vertices']

    # first we extract all vertices that are part of the ROI
    # and then we filter the faces to only include those that are part of the ROI
    if roi_mask is not None:
        vertices = vertices[roi_mask]
        faces = faces[np.all(roi_mask[faces], axis=1)]
        
    # Find boundary edges by creating edges from faces then counting occurrences
    # Non boundary edges will appear twice in the edge list
    # while boundary edges will appear only once
    edges = np.vstack([faces[:, [0, 1]],
                       faces[:, [1, 2]],
                       faces[:, [2, 0]]])
    # Sort the vertices in each edge to standardize, then find unique edges
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

    # Return edges that appear only once
    boundary_edges = unique_edges[counts == 1]
    return boundary_edges

def plot_edges(edges, surf, ax=None, outline_color='grey', outline_width=0.5):    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # Plot only the boundary edges
    for i, j in edges:
        ax.plot([surf['vertices'][i, 0], surf['vertices'][j, 0]],
                [surf['vertices'][i, 1], surf['vertices'][j, 1]],
                color=outline_color, linewidth=outline_width)

def plot_surf_atlas_boundaries(atlas_fname, surf, ax=None):
    """ Plots the outlines of the atlas regions on the surface.
    atlas is a gifti label file. It contains a region number for each vertex,
    as well as a list of region names and colors.
    surf is assumed to be a flat left hemisphere surface.
    We first plot the surface as gray in 2D and its outline in black.
    Then we plot all atlas regions as colored outlines.
    """
    global_alpha = 0.5
    # load atlas GIFTI label file, get the data, labels, indices and colors
    atlas_gifti = nib.load(atlas_fname)
    data = atlas_gifti.darrays[0].data
    label_colors = dict()
    for label in atlas_gifti.labeltable.labels:
        label_colors[label.key] = (label.red, label.green, label.blue, label.alpha * global_alpha)
    label_names = atlas_gifti.labeltable.get_labels_as_dict()
    label_names = {k: v.replace('_ROI', '') for k, v in label_names.items()}
    
    if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

    for label_idx in label_colors.keys():
        # create a mask for the current label
        roi_mask = (data == label_idx)
        # find the boundary edges of the current label
        boundary_edges = find_boundary_edges(surf, roi_mask=roi_mask)
        # plot the edges of the current label
        plot_edges(boundary_edges, surf, ax=ax, outline_color=label_colors[label_idx], outline_width=0.5)

    return ax

def plot_surf_data_left_hemi(data, surf, atlas=None, ax = None, vmin = None, vmax = None, cmap=None, 
                             zero_as_transparent=False):
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.max(data)

    outline_color = 'grey'
    outline_width = 0.5
    if cmap is None:
        cmap = plt.cm.hot
    flatsurf = Triangulation(surf['vertices'][:, 0], surf['vertices'][:, 1], surf['faces'])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # first plot the surface as uniform gray background then plot the data on top
    ax.tripcolor(flatsurf, np.ones_like(data) * 15, cmap=plt.cm.tab20, shading='gouraud', vmin=0, vmax=20)
    if zero_as_transparent:
        data[data == 0] = np.nan
    tpc = ax.tripcolor(flatsurf, data, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    

    ax.set_aspect('equal')
    ax.axis('off')

    # Find boundary edges
    boundary_edges = find_boundary_edges(surf)
    plot_edges(boundary_edges, surf, ax, outline_color, outline_width)

    # plot atlas parcel outlines if atlas is provided
    if atlas is not None:
        plot_surf_atlas_boundaries(atlas, surf, ax=ax)


    return tpc

def plot_surf_clusters_left_hemi(data, surf, atlas=None, ax=None, vmax=None, 
                                 label_idcs=None, label_dict=None, label_fontsize=8):
    cmap = plt.cm.tab20([15,0,2,4,6,8,10,12,16,18,1,3,5,7,9,11,13,17,19])
    cmap = plt.cm.colors.ListedColormap(cmap)

    no_cluster = (data == 0)
    clusters = np.mod(data-1,18)+1
    clusters[no_cluster] = 0
    tpc = plot_surf_data_left_hemi(clusters, surf, atlas=atlas, ax=ax, vmax=18, cmap=cmap)
    ax = tpc.axes
     # optionally we add an annotation that labels every cluster with its index
    # to do so we find the center of mass of each cluster and use it to place the text
    if label_dict is not None:
        for idx in label_dict.keys():
            if idx != 0:
                cluster_mask = (data == idx)
                if np.any(cluster_mask):
                    cluster_vertices = surf['vertices'][cluster_mask]
                    center_of_mass = np.mean(cluster_vertices, axis=0)
                    ax.text(center_of_mass[0], center_of_mass[1], label_dict[idx], color='black', fontsize=label_fontsize,
                            ha='center', va='center')

    else:
        for i in range(0, 18 + 1):
            if i != 0:
                cluster_mask = (data == i)
                if np.any(cluster_mask):
                    cluster_vertices = surf['vertices'][cluster_mask]
                    center_of_mass = np.mean(cluster_vertices, axis=0)
                    ax.text(center_of_mass[0], center_of_mass[1], str(i), color='black', fontsize=8,
                            ha='center', va='center')
    
    return tpc
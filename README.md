# Exploring Academic Relationships with UMAP and OpenAlex

GitHub Repository for my Master's Thesis project: **Exploring Academic Relationships with UMAP: Dimensionality Reduction and Visualization of Topics and Authors in OpenAlex**

Supervisors: Dimitri Marinelli and Albert Diaz-Guilera.

## 📊 Motivation and Overview
In an era of rapidly growing scientific publications, making sense of vast academic knowledge and uncovering meaningful relationships between research topics and authors is increasingly important.

This thesis applies Uniform Manifold Approximation and Projection (UMAP) to analyse and visualize research works from the OpenAlex database. By using various embedding methods (including transformer-based models and hierarchical topic encodings) the study demonstrates that UMAP projections can effectively capture meaningful structures in the data, revealing relationships among research areas and institutions. 

Results show that capturing complex topic relationships across multiple domains is a challenging task. Nevertheless, the visualizations reveal significant thematic clusters and author groupings that align with our data analysis. Quantitative evaluation using clustering metrics, such as the silhouette score, confirms the agreement between visual patterns and semantic embeddings. We also show the impact of UMAP hyperparameters on balancing local and global data structure preservation, which influences visualization clarity and interpretability. 

The resulting interactive, zoomable visual maps provide researchers with a powerful tool to explore and understand the organization of scientific knowledge.

## 🔗 Interactive Website
Explore the interactive visualizations here: [Interactive Visualization](https://albagarciaromo.github.io/)


## 🗂 Repository Structure

This repository contains the full codebase and outputs of the project.

### `code/`
- Contains 7 Jupyter notebooks covering the different aspects of the thesis, roughly aligned with the thesis chapters.
- Includes `utils.py`: a Python module with all the utility functions used across the notebooks, cleaned and ready for reuse.

### `figures/`
- All figures generated throughout the project and used in the thesis.

### `interactive_visualizations/`
- Interactive HTML visualizations created as part of the analysis.

---

Feel free to clone, explore, or reuse the code and visualizations under the terms of the license.

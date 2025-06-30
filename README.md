# Exploring Academic Relationships with UMAP and OpenAlex

GitHub Repository for my Master's Thesis project: **Exploring Academic Relationships with UMAP: Dimensionality Reduction and Visualization of Topics and Authors in OpenAlex**

Supervisors: Dimitri Marinelli and Albert Diaz-Guilera.

## Abstract
This thesis applies Uniform Manifold Approximation and Projection (UMAP) to
analyse and visualize research works from the OpenAlex database. By using various
embedding methods (including transformer-based models and hierarchical topic
encodings) the study demonstrates that UMAP projections can effectively capture
meaningful structures in the data, revealing relationships among research areas and
institutions. Results show that capturing complex topic relationships across multi-
ple domains is a challenging task. Nevertheless, the visualizations reveal significant
thematic clusters and author groupings that align with our data analysis. Quan-
titative evaluation using clustering metrics, such as the silhouette score, confirms
the agreement between visual patterns and semantic embeddings. We also show
the impact of UMAP hyperparameters on balancing local and global data structure
preservation, which influences visualization clarity and interpretability. The result-
ing interactive, zoomable visual maps provide researchers with a powerful tool to
explore and understand the organization of scientific knowledge.

## 🔗 Interactive Website
Explore the interactive visualizations here: [https://albagarciaromo.github.io/](Interactive Visualizations)



## Introduction
This project addresses the challenge of understanding and exploring the structure of academic knowledge through data visualization. Using the OpenAlex database, we develop an interactive tool that reveals relationships between research works and authors. The approach combines text embeddings (from titles and abstracts) with UMAP dimensionality reduction to produce interpretable 2D maps. The project experiments with various embedding models and UMAP configurations, focusing on two institutions, University of Barcelona and Utrecht University, to uncover patterns across topics and collaboration networks.

## Conclusions
The project demonstrates that UMAP combined with semantic embeddings can effectively visualize topic and author relationships in scholarly data. The resulting visualizations align well with expected domain structures and clustering metrics. Author maps also revealed research communities and potential collaboration paths. Limitations include UMAP's local focus and OpenAlex topic label accuracy. Future improvements include connecting the tool to the OpenAlex API, incorporating secondary topic data, analyzing author trajectories over time, and ensuring visualization stability.

## 🗂 Repository Structure

This repository contains the full codebase and outputs of the project.

### `code/`
- Contains 7 Jupyter notebooks covering the different aspects of the thesis, roughly aligned with the thesis chapters.
- Includes `util.py`: a Python module with all the utility functions used across the notebooks, cleaned and ready for reuse.

### `figures/`
- All figures generated throughout the project and used in the thesis.

### `interactive_visualizations/`
- Interactive HTML visualizations created as part of the analysis.

---

Feel free to clone, explore, or reuse the code and visualizations under the terms of the license.

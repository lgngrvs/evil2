# Evil

This is a research repository building on the literature in emergent misalignment. We are particularly grateful to Soligo and Turner for their [open-source models and high-quality research](https://github.com/clarifying-EM/model-organisms-for-EM/tree/main), as well as Andy Arditi for providing [the SAEs we use](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct) for Qwen2.5 7B Instruct.

Currently, `evil` can: 
- Replicate the convergent misalignment direction discovered in [Soligo et al. 2025](https://arxiv.org/pdf/2506.11618) in Qwen 2.5 7B Instruct
- Use PCA to acquire a different misalignment direction vector
- Given sparse autoencoders for Qwen 2.5 7B Instruct, identify SAE features most changed when steering with that misalignment direction and use an LLM to label them (not highly robust, but can detect meaningful signal) 
- Evaluate similarity metrics between SAE features, cosine similarity, and PCA steering vectors
- Run controls (steering on random vector, cosine similarity with random vector)
- Generate plots for all of these things

Everything can be run using the Jupyter Notebook in `colab.ipynb`. (We built for colab because colab is currently the easiest way for us to access compute, sorry)

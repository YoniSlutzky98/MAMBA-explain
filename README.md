# MAMBA-explain
A MAMBA explainability final project done in the deep learning and advanced topics course @ school of Electrical 
Engineering, TAU.  

## Installing Requirements

Tested with Python 3.9. The ```requirements.txt``` file includes the required dependencies, which can be installed via:

```
pip install -r requirements.txt
```

## Experiments

All the experiments were performed within jupyter notebooks, with the main functionality being imported from the 
```mamba2mini.py``` module. The experiments were carried out using a single Nvidia RTX A6000 GPU. All experiments 
attempt to use a GPU.

### Mamba2 Model

Throughout our experiments, the model used is the ```mamba2-1.3b``` model 
([Dao et al. 24'](https://arxiv.org/abs/2405.21060)) whose weights are imported from 
[huggingface](https://huggingface.co/state-spaces/mamba2-1.3b). The tokenizer used is the ```gpt-neox-20b``` tokenizer 
([Black et al. 22'](https://arxiv.org/abs/2204.06745)) imported from 
[huggingface](https://huggingface.co/EleutherAI/gpt-neox-20b).  

We adapt the ```mamba2mini.py``` module found in [this repo](https://github.com/tommyip/mamba2-minimal). The main 
modification made was to allow the model to reconstruct the underlying linear attention layer and compute the predictions
according to it, instead of the regular inference procedure. For further details on the attention structural property of 
Mamba2, see [Dao et al. 24'](https://arxiv.org/abs/2405.21060).

### Data Organization

The original dataset used is ```counterfact-tracing``` 
(available [here](https://huggingface.co/datasets/NeelNanda/counterfact-tracing)), and it is saved in 
```originl_data.parquet```. We extract the prompts for which the model is correct before beginning our experiments. For
control purposes, these prompts are extracted for both the original model (i.e., without attention) and for the attention 
model. The following table details which notebook extracts data for which model:

| Model           | Notebook                                |
|-----------------|-----------------------------------------|
| Original Model  | ```data_construction_original.ipynb```  |
| Attention Model | ```data_construction_attention.ipynb``` |

For each notebook, running all of its cells in order produces a parquet file containing the prompts, the predictions 
made by the model, and the probabilities the model gave for the true token and for the predicted token. We ensure the 
outputs are identical in ```original_attention_comparison.ipynb```.

### Experiment 1 - Info Flow Analysis

The experiment uses the attention model to predict the next token for correct prompts of the core model, while blocking 
attention to the last token over continuous windows of layers from either the second to last token, the ```subject``` 
tokens or the ```relation``` tokens. Both the prediction probability of the true token and the overall accuracy are 
tracked. To run the experiments, run all cells of ```info_flow_experiments.ipynb``` in order. To plot the results 
obtained, run all cells of ```info_flow_results.ipynb``` in order. This would create the plots seen in the document 
(probability plots are akin to the ones in section 5 of [Geva et al. 23'](https://arxiv.org/abs/2304.14767)). 

### Experiment 2 - Heatmap Analysis

The experiment uses the attention model to predict the next token for correct prompts of the core model, while blocking 
attention to the last token over continuous windows of layers from each individual token. The results are plotted in
heatmaps akin to the ones in appendix H of [Geva et al. 23'](https://arxiv.org/abs/2304.14767). To run the experiments, 
run all cells of ```heatmap_experiment.ipynb``` in order. 
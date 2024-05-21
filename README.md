# abliterator.py
Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens

This library is very barebones for right now, and documentation is slim at the moment. I want to publish this now and hopefully bring this up to snuff over time, with the help of the community.

Right now, this works very well for my own personal workflow, but I would like to automate this, and broaden out from the pure "harmless / harmful" feature ablation, to augmentation, and adding additional features

Documentation will be up 

To load a model in:
```python
import abliterator

model = "LoadedInModel/Llama-3-70B-Instruct-but-Better"  # the huggingface or path to the model you're interested in loading in
aliased_model = "meta-llama/Meta-Llama-3-70B-Instruct"   # load in model using the config from its base model -- makes transformer lens happy
device = 'cuda'                             # optional: defaults to cuda
n_devices = None                            # optional: when set to None, defaults to `device.cuda.device_count`
cache_fname = 'my_cached_point.pth'         # optional: if you need to save where you left off, you can use `save_activations(filename)` which will write out a file. This is how you load that back in.
activation_layers = None                    # optional: defaults to ['resid_pre', 'resid_mid', 'resid_post'] which are the residual streams. Setting to None will cache ALL activation layer types
chat_template = None                        # optional: defaults to Llama-3 instruction template. You can use a format string e.g. ("<system>{instruction}<end><assistant>") or a custom class with format function -- it just needs an '.format(instruction="")` function. See abliterator.ChatTemplate for a very basic structure.
negative_toks = [4250]                      # optional, but highly recommended: ' cannot' in Llama's tokenizer. Tokens you don't want to be seeing. Defaults to my preset for Llama-3 models
positive_toks = [23371, 40914]              # optional, but highly recommended: ' Sure' and 'Sure' in Llama's tokenizer. Tokens you want to be seeing, basically. Defaults to my preset for Llama-3 models

abliterator.ModelAbliterator(
  model,
  dataset,
  aliased_model,
  device,
  n_devices=None,
  cache_fname=None,
  activation_layers=['resid_pre', 'resid_mid', 'resid_post'],
  chat_template=None,
  positive_toks={},
  negative_toks={}
)
```

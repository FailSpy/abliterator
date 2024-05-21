# abliterator.py
Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens

This library is so exceedingly barebones for right now, and documentation is slim at the moment. Right now, it feels like a glorified IPython notebook.
I want to publish this now and hopefully bring this up to snuff over time, with the help of the community.

Right now, this works very well for my own personal workflow, but I would like to automate this further, and broaden out from the pure "harmless / harmful" feature ablation, to augmentation, and adding additional features

## Loading a model in
```python
import abliterator

model = "LoadedInModel/Llama-3-70B-Instruct-but-Better"  # the huggingface or path to the model you're interested in loading in
dataset = [abliterator.get_harmful_instructions(), abliterator.get_harmless_instructions()] # datasets to be used for caching and testing, split by harmful/harmless
aliased_model = "meta-llama/Meta-Llama-3-70B-Instruct"   # optional: load in model using the config from its base model -- makes transformer lens happy. Can be None
device = 'cuda'                             # optional: defaults to cuda
n_devices = None                            # optional: when set to None, defaults to `device.cuda.device_count`
cache_fname = 'my_cached_point.pth'         # optional: if you need to save where you left off, you can use `save_activations(filename)` which will write out a file. This is how you load that back in.
activation_layers = None                    # optional: defaults to ['resid_pre', 'resid_mid', 'resid_post'] which are the residual streams. Setting to None will cache ALL activation layer types
chat_template = None                        # optional: defaults to Llama-3 instruction template. You can use a format string e.g. ("<system>{instruction}<end><assistant>") or a custom class with format function -- it just needs an '.format(instruction="")` function. See abliterator.ChatTemplate for a very basic structure.
negative_toks = [4250]                      # optional, but highly recommended: ' cannot' in Llama's tokenizer. Tokens you don't want to be seeing. Defaults to my preset for Llama-3 models
positive_toks = [23371, 40914]              # optional, but highly recommended: ' Sure' and 'Sure' in Llama's tokenizer. Tokens you want to be seeing, basically. Defaults to my preset for Llama-3 models

my_model = abliterator.ModelAbliterator(
  model,
  dataset,
  aliased_model,
  device='cuda',
  n_devices=None,
  cache_fname=None,
  activation_layers=['resid_pre', 'resid_mid', 'resid_post'],
  chat_template=None,
  positive_toks={},
  negative_toks={}
)
```

## Cache activations/sample dataset
Once loaded in, run the model against N samples of harmful, and N samples of harmless so it has some data to work with:
```python
my_model.cache_activations(N=512,reset=True,preserve_harmless=True)
```
`preserve_harmless=True` is generally useful, as it keeps the "desired behaviour" unaltered from any stacked modifications if you run it after some mods.

## Getting refusal directions from the cached activations
Speaking of modding, here's a simple representation of how to pick, test, and actually apply a direction from a layer's activations:
```python
refusal_dirs = my_model.refusal_dirs()
testing_dir = refusal_dirs['blocks.18.hook_resid_pre']
my_model.test_dir(testing_dir, N=32, use_hooks=True) # I recommend use_hooks=True for large models as it can slow things down otherwise, but use_hooks=False can give you more precise scoring to an actual weights modification
```
`test_dir` will apply your refusal_dir to the model temporarily, and run against N samples of test data, and return a composite (negative_score, positive_score) from those runs. Generally, you want negative_score to go down, positive_score to go up.

### Testing lots of refusal directions

This is one of the functions included in the library, but it's also useful for showing how this can be generalized to test a whole bunch of directions.
```python
    def find_best_refusal_dir(N=4, use_hooks=True, invert=False):
        dirs = self.refusal_dirs(invert=invert)
        scores = []
        for direction in tqdm(dirs.items()):
            score = self.test_dir(direction[1],N=N,use_hooks=use_hooks)[0]
            scores.append((score,direction))
        return sorted(scores,key=lambda x:x[0])

```

## Applying the weights

And now, to apply it!
```python
my_amazing_dir = find_best_refusal_dir()[0]
my_model.apply_refusal_dirs([my_amazing_dir],layers=None)
```
Note the `layers=None`. You can supply a list here to specify which layers you want to apply the refusal direction to. None will apply it to all writable layers.

### Blacklisting specific layers
Sometimes some layers are troublesome no matter what you do. If you're worried about accidentally replacing it, you can blacklist it to prevent any alteration from occurring:
```python
my_model.blacklist_layer(27)
my_model.blacklist_layer([i for i in range(27,30)]) # it also accepts lists!
```

#### Whitelisting
And naturally, to undo this and make sure a layer can be overwritten:
```
my_model.whitelist_layer(27)
```
By default, all layers are whitelisted.

Neither of these will provide success/failure states. They will just assure the desired state in running it at that instant.

## Confirming your model is A-okay
Now to make sure you've not damaged the model dramatically after applying some stuff, you can do a test run:
```python
with my_model: # loads a temporary context with the model
  ortho.apply_refusal_dir([my_new_precious_dir]) # Because this was applied in the 'with my_model:', it will be unapplied after coming out.
  print(my_model.mse_harmless(N=128)) # While we've got the dir applied, this tells you the Mean Squared Error using the current cached harmless runs as "ground truth" (loss function, effectively)
```

### Want to see it run? Test it!
```python
ortho.test(N=16,batch_size = 4) # runs N samples from the harmful test set and prints them for the user. Good way to check the model hasn't completely derailed.
# Note that by default if a test run produces a negative token, it will stop the whole batch and move on to the next. (it will show lots of '!!!!' in Llama-3's case, as that's token ID 0)

ortho.generate("How much wood could a woodchuck chuck if a woodchuck could chuck wood?") # runs and prints the prompt!
```

## Utility functions
Documentation coming soon.

## How to Save as a HuggingFace model
Functionality coming soon. For now, use PyTorch's saving method, or see my notebook for an idea of how to do this yourself.



# Create scorings
Start a jupyter-notebook server in the project root and open the `FEIII_final_scoring.ipynb` file ans start executing the cells.
Within the `pipeline()` and `score_func()` functions, you can adjust which classifier and scoring function to use by setting the variable directly after the function header.

In case the train/eval split isn't working out, just shuffle the sampling by executing
```
data.shuffle_train_eval(n_docs_eval=3, max_tries=5)
```

Remember to adjust the `classifier` variable and scoring function accordingly before executing the cell that saves the frame(s).

# Train and save embeddings
```
$ cd <project_root>/
$ python
>>> from code import feiii_transformers as ft
>>> e = ft._EmbeddingHolder('<path_to_full_repots>')
>>> e.train(num_files=30, num_epochs=25)
reading: ALLY_2016.html
reading: ALLY_2014.html
reading: CAPITAL-ONE_2013.html
...
extracting sentences...
words: 1630611
sentences: 56085
Training-Epoch: 0 | lr: 0.025
...
>>> e.save('<path_to_embedding>')
```


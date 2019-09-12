# Reinforcement learning for connect 4

### Train agent (self-play):

```bash
python Training.py
```

### Evaluate performance

Evaluate the learning progress by letting the agent play against earlier versions of itself. Run

```bash
python evaluate_checkpoints.py
```

### TO DO

* Try deeper network (adding convolutional layers)

Implemented, but does not work because it seems that the network is overfitting very quickly on one column, then if that column is full, we set the probability to zero, then our clipping and normalizing stuff (which is also just a workaround) does not work anymore since we get log(0) and therefore nan values etc. 

Possible solutions: Avoid overfitting (previously it did work because we the agent did not have such high probabilities for any column), change clipping process (I have tried clipping at different values and normalize afterwards again, but so far without success).

* evaluate after each epoch of training instead of in the end

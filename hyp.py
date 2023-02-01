# lr
lr0: 0.0001
warmup_lr: 0.00001   
warm_epoch:5


# setting
num_classes: 20    # TODO: This must be changed!

# training
epochs: 100
batch_size: 8      # TODO amp:8 otherwise 2|4
save_interval: 5
test_interval: 5

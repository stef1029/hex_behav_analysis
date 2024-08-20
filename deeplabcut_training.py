config = r'/beegfs3/srogers/DLC-Project-mtap10-1f - stim-06232023163325-0000-SRC-2023-07-03/config.yaml'
# add perhaps config path as input function

import deeplabcut
import os

superanimal_name = "superanimal_topviewmouse"


# Create Training dataset
print("Creating Training Dataset")
deeplabcut.create_training_dataset(config,  
                                   net_type='resnet_50', 
                                   augmenter_type='imgaug', 
                                   windows2linux = True,
                                   superanimal_name = superanimal_name)
print("Training Dataset created")


# train
print("Training network")
deeplabcut.train_network(config, 
                         gputouse=os.environ.get("CUDA_VISIBLE_DEVICES"), 
                         maxiters=800000,
                         superanimal_name = superanimal_name,
                         superanimal_transfer_learning = True)
print("Network Trained")

# Evaluate
print("Evaluating network")
deeplabcut.evaluate_network(config, 
                            plotting=False, 
                            gputouse=os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Network evaluated")



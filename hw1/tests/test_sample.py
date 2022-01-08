import numpy as np
import torch
from hw1.load_data import DataGenerator

# Make sure Load_date works a expected
def test_load_data():

    data_generator = DataGenerator(
        num_classes = 3, num_samples_per_class = 7, device='cpu'
    )

    train, labels = data_generator.sample_batch('train', 4)
    assert train.shape == (4,8,3,784)
    assert labels.shape == (4,8,3,3)

    # check same tensor
    # expected_output_for_batch_3 = torch.tensor(
    #                 [[[1., 0., 0.],
    #                 [0., 1., 0.],
    #                 [0., 0., 1.]],

    #                 [[1., 0., 0.],
    #                 [0., 1., 0.],
    #                 [0., 0., 1.]],

    #                 [[0., 0., 1.],
    #                 [1., 0., 0.],
    #                 [0., 1., 0.]]])
    
    # breakpoint()
    # assert torch.all( torch.eq(labels[3], expected_output_for_batch_3) )





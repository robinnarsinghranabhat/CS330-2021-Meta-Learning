## Assignment 1 : 
- Navigate inside `hw1` directory
- Sample Execute : `python hw1.py --model_size 256 --model_type "DNC_experiment" --meta_batch_size 256`
- NOTE : To train with just LSTM model, don't include string "DNC" inside `--model_type` argument. <br> Example : `python hw1.py --model_size 256 --model_type "DNC_experiment" --meta_batch_size 256`

    Best Model Artifacts Saved at: 
    - hw1/runs : Logs for tensorboard
    - hw1/trained_models : Latest trained model
- If the script is re-run with the same arguments, training will resume from last checkpoint, given that model is saved.
- For viewing results in Tensorboard, execute in the terminal : <br>
    `tensorboard --logdir="PATH_TO_LOG_DIRECTORY"`
    
    Example command :  `tensorboard --logdir=".\hw1\runs\2022-01-09\LSTM_experiment\K_10_N_5_B_256_H_256"`

- RESULTS:
    - DNC-Model
        - Meta-Test Accuracy <img src=".\result_plots\DNC_model_test_accuracy.PNG"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" /> <br>
        - Train and Test Loss <img src=".\result_plots\DNC_model_train_and_test_loss.PNG"
     alt="Markdown Monster icon" /> <br> 

    - LSTM-Model
        - Meta-Test Accuracy <img src=".\result_plots\DNC_model_test_accuracy.PNG"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" /> <br>
        - Train and Test Loss <img src=".\result_plots\DNC_model_train_and_test_loss.PNG"
     alt="Markdown Monster icon" /> <br> 

## Additonal NOTES FOR training with [colab-ssh](https://github.com/WassimBenzarti/colab-ssh)

(**this may not work and just for experimental purpose**) <br>
If you are running the above script from through colab-ssh :
- First Mount Google drive into your colab-vm machine 
- Set `--colab_mode 1`,  to save model configs in google drive while executing. And update the path inside : `save_to_colab` function
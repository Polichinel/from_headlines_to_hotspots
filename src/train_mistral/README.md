Training and prediction files for Mistral 7B LLM. 

`train_sloth` uses unsloth to fine-tune the model with a corpus spanning month 425-460 using simple prompts and RAG.
`predict_sloth` uses unsloth to predict directly from the second stage model
`train_predict_sloth` uses unsloth to fine-tune the fine-tune the previous stage with m-[2,1] and predict for m (with targets m, m+1, m+3 and m+6)
`sh` files are used to control the behavior of Chalmers' Alvis HPC and set and run jobs.

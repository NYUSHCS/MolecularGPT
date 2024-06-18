# MolecularGPT
### This is the code for paper MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction


#### Test the performance on classification tasks 
CUDA_VISIBLE_DEVICES=0 python downstream_test_llama_cla.py \
    --load_8bit \
    --base_model /home/leslie/Llama-2-7b-chat-hf \

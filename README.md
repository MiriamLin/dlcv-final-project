[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/aUssIOLl)
# DLCV Final Project ( Warehouse Spatial Intelligence )

# Notice
1. Please write your readme file clearly.
2. Please write your README clearly, including step-by-step instructions on how to run your code.
3. Please click [this link](https://docs.google.com/presentation/d/1wbZBPT7-C7j0wt5RqpNxZMWoqZKYw1fdizRz9hojfeQ/edit?slide=id.g3196368f9ef_0_0#slide=id.g3196368f9ef_0_0) to view the slides of Final Project - Warehouse Spatial Intelligence. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

## 🔧 Installation
1. Create and activate a conda environment with Python 3.13
```
conda create -p .conda python=3.13 -y
conda activate ./.conda
```
2. Install Python dependencies
```
pip3 install -r requirements.txt
```

## Inference with Pre-trained Model
1. Download the pre-trained model from Google Drive:
```
bash download_ckpt.sh
```
2. Run inference code:
```
python3 alg.py --json_file final_data/test.json --data_dir final_data/test --prediction_file predictions.json --dist_config_path dist_config.json --inclusion_config_path inclusion_config.json
```
Replace `final_data/test.json` and `final_data/test` with your own test set path if needed.

## Train Your Own Model
1. Train the distance estimation model:
```
python3 organize_dist_est.py --input_file final_data/train.json --output_file train_dist_est.json
python3 agent/distance_est/train.py --data_dir final_data/train --json_file train_dist_est.json --ckpt_dir distance_ckpt --no_rgb --log_scale --num_epochs 11
```

2. Train the inclusion estimation model:
```
python3 organize_inside.py --input_file final_data/train.json --output_file train_inside.json
python3 agent/inside_pred/train.py --save_path inclusion_ckpt --use_rgb --epochs 11 --json train_inside.json --image_dir final_data/train/images/ --depth_dir final_data/train/depths/
```

# Dataset 
Please click [this link](https://huggingface.co/datasets/yaguchi27/DLCV_Final1?fbclid=IwY2xjawOaLR9leHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeocpm74PJ10FFgl66Byw0iJksQVcHBSvJ1nJMCXAKDltmAfpIS-k4kMzPYWw_aem_U6QUG06tJIJ--JHPvPxUhA) and follow the instruction to download the dataset.

# Submission Rules
### Deadline
114/12/22 (Mon.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntu-dlcv-2025-fall-ta@googlegroups.com](mailto:ntu-dlcv-2025-fall-ta@googlegroups.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion



# # 1. save_um_embedding_predict_before
# CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-05-1725 --load_before_or_after before --load_um_or_uram um --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# # 2. save_uram_embedding_predict_before
# CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-05-1725 --load_before_or_after before --load_um_or_uram uram --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# # 3. save_uram_embedding_predict_after
# CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-05-1725 --load_before_or_after after --load_um_or_uram uram --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# # 4. save_uram_embedding_predict_after_abl2  (save_dir 改为 ./outputs-after-step05-abl2)
# CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-step05-ablation-2-0812-1015 --load_before_or_after after --load_um_or_uram uram --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# 1. save_um_embedding_predict_before
CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-ft04-pm-1623 --load_before_or_after before --load_um_or_uram um --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# 2. save_uram_embedding_predict_before
CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-ft04-pm-1623 --load_before_or_after before --load_um_or_uram uram --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# 3. save_uram_embedding_predict_after
CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-ft04-pm-1623 --load_before_or_after after --load_um_or_uram uram --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

# 4. save_uram_embedding_predict_after_abl2  (save_dir 改为 ./outputs-after-step05-abl2)
# CUDA_VISIBLE_DEVICES=1 python save_predicts_embedding.py --save_dir /nvme/szh/data/3ai/lips/outputs-after-step05-ablation-2-0812-1015 --load_before_or_after after --load_um_or_uram uram --save_data_path /nvme/szh/data/3ai/lips/saved_data_embedding

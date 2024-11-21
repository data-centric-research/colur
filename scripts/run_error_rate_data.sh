# CIFAR-10
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.1 --noise_type symmetric --dataset cifar-10
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.25 --noise_type symmetric --dataset cifar-10
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.5 --noise_type symmetric --dataset cifar-10
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.75 --noise_type symmetric --dataset cifar-10
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.9 --noise_type symmetric --dataset cifar-10

# FLOWER-102
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.1 --noise_type symmetric --dataset flower-102

# rerun
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.25 --noise_type symmetric --dataset flower-102

python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.5 --noise_type symmetric --dataset flower-102
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.75 --noise_type symmetric --dataset flower-102
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.9 --noise_type symmetric --dataset flower-102

# CIFAR-100
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.1 --noise_type asymmetric --dataset cifar-100
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.25 --noise_type asymmetric --dataset cifar-100
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.5 --noise_type asymmetric --dataset cifar-100
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.75 --noise_type asymmetric --dataset cifar-100
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.9 --noise_type asymmetric --dataset cifar-100

# PET-37
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.1 --noise_type asymmetric --dataset pet-37
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.25 --noise_type asymmetric --dataset pet-37
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.5 --noise_type asymmetric --dataset pet-37
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.75 --noise_type asymmetric --dataset pet-37
python ./result_analysis/gen_result_visualize_to_csv.py --noise_rate 0.9 --noise_type asymmetric --dataset pet-37

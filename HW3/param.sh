# 用于超参数对比
rm -r -f param_exp
mkdir param_exp
dir_path="param_exp"
for bsz in 16 32 64
do
    for lr in 1 0.1 0.01 0.001
    do
        comb="${bsz}-${lr}"
        mkdir "${dir_path}/${comb}"
        working_dir="${dir_path}/${comb}"
        nohup python -u 2022210870.py --num_classes 50 --num_samples_train 15 --num_samples_test 5 --model cnn  --split 0.2 --bsz "${bsz}" --lr "${lr}" --path "${working_dir}" \
            > "${working_dir}/${comb}.out" 2>&1 & echo $! > "${working_dir}/${comb}.pid"
    done
done
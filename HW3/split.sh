# 不同数据集划分效果对比
rm -r -f split_exp
mkdir split_exp
dir_path="split_exp"
for classes in 25 50 100
do
    for train in 5 10 15 18
    do
        comb="${classes}-${train}"
        mkdir "${dir_path}/${comb}"
        working_dir="${dir_path}/${comb}"
        nohup python -u main.py --num_classes "${classes}" --num_samples_train "${train}" --num_samples_test "$((20-train))" --model cnn  --split 0.2 --bsz 32 --lr 0.1 --path "${working_dir}" \
            > "${working_dir}/${comb}.out" 2>&1 & echo $! > "${working_dir}/${comb}.pid"
    done
done
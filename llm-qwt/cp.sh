#!/bin/bash

start_time=$(date +%s)  # 记录开始时间
# -------------------------------------------------------------------------
cpdir_with_progress() {
    local src_dir="$1"
    local dst_dir="$2"

    if [[ ! -d "$src_dir" ]]; then
        echo "❌ Source directory '$src_dir' does not exist."
        return 1
    fi

    mkdir -p "$dst_dir"

    echo "📂 Copying all files from '$src_dir' to '$dst_dir'..."

    find "$src_dir" -type f | while read -r src_file; do
        # 保留原始文件夹结构
        rel_path="${src_file#$src_dir/}"
        dst_file="$dst_dir/$rel_path"
        dst_subdir=$(dirname "$dst_file")
        mkdir -p "$dst_subdir"

        # 显示 tqdm 进度
        if [[ -f "$src_file" ]]; then
            size=$(stat -c %s "$src_file")
            echo "📦 Copying '$rel_path'  [$(numfmt --to=iec $size)]"
            cat "$src_file" | tqdm --bytes --total "$size" > "$dst_file"
        fi
    done
}

# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/opt-125m /localssd/lbxj/opt-125m
# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/opt-1.3b /localssd/lbxj/opt-1.3b
# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/opt-2.7b /localssd/lbxj/opt-2.7b
# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/opt-6.7b /localssd/lbxj/opt-6.7b
# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/opt-13b /localssd/lbxj/opt-13b

# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/TinyLlama-1.1B-Chat-v1.0 /localssd/lbxj/TinyLlama-1.1B-Chat-v1.0
# cpdir_with_progress /cephfs/shared/wangzw/LLM_models/llama-2-7b-hf /localssd/lbxj/llama-2-7b-hf

cpdir_with_progress /cephfs/shared/wangzw/LLM_models/Qwen3-8B /localssd/lbxj/Qwen3-8B
# cpdir_with_progress /cephfs/shared/juxin/models/Qwen3-1.7B /localssd/lbxj/Qwen3-1.7B
# -------------------------------------------------------------------------
end_time=$(date +%s)  # 记录结束时间
duration=$((end_time - start_time))
# 格式化输出为小时-分钟-秒
printf "RUNNING TIME: %02dh-%02dm-%02ds\n\n" $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))

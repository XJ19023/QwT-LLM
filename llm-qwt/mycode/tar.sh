#!/bin/bash
start_time=$(date +%s)  # 记录开始时间
# -------------------------------------------------------------------------
# 1. 设置变量
SOURCE_DIR="QwT-LLM"        # 当前用户的 home 目录
BACKUP_DIR="/cephfs/juxin/backup"               # 修改为你想保存备份的目录
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S") # 当前时间戳
BACKUP_NAME="backup_${TIMESTAMP}.tar.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# 备份说明
comment="\
1. per group 分布均匀，没办法压缩.\\n\
1. 不能跑batch 可能是问题所在.\\n\
2. next step: 手动进行batch循环收集数据.\\n\
---------------------------------------\\n\
1. opt-125m quant qwt good.\\n\
2. next step: check OOM.\\n\
---------------------------------------\\n\
1. opt quant(per channel clamp, per group quant) qwt good.\\n\
2. now: run opt per group quant qwt.\\n\
3. next step: implement llama qwt .\\n\
---------------------------------------\\n\
1. opt quant(per channel clamp, per group quant) qwt(align to full/compensate) good.\\n\
3. next step: implement llama qwt .\\n\
"


# 2. 创建备份目录（如果不存在）
mkdir -p "$BACKUP_DIR"

# 3. 执行打包（忽略缓存等大文件夹，可自定义）
echo "back up from ${SOURCE_DIR} to ${BACKUP_PATH} ..."
cd /cephfs/juxin/git_test ; \
echo -e "${comment}" > "${SOURCE_DIR}/backup.log"
tar --exclude="${SOURCE_DIR}/.cache" \
    --exclude="${SOURCE_DIR}/.local/share/Trash" \
    -czvf "$BACKUP_PATH" "$SOURCE_DIR"

rm "${SOURCE_DIR}/backup.log"
cd -
# 4. 完成提示
if [[ $? -eq 0 ]]; then
    echo "back up complete at: $BACKUP_PATH"
else
    echo "back up failed"
fi

# -------------------------------------------------------------------------
end_time=$(date +%s)  # 记录结束时间
duration=$((end_time - start_time))
# 格式化输出为小时-分钟-秒
printf "RUNNING TIME: %02dh-%02dm-%02ds\n\n" $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))

# python m6doc_classify.py
# 定义数据集名称数组
# datasets=("D4LA" "ElemLayout" "M6Doc" "DocLayNet" )
datasets=("D4LA")
# datasets=("DocLayNet")
# datasets=("M6Doc")
output_dir="/workspace/datasets/layout/unsv2_layout_ls_v2"
# 如果 output_dir 不存在，则创建, 如果存在则删除
if [ ! -d "$output_dir" ]; then 
    mkdir -p "$output_dir"
fi
# else
#     rm -rf "$output_dir"
#     mkdir -p "$output_dir"
# fi

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    # 转换为小写用于输出文件名
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    
    # 处理训练集
    python label_studio_converter.py \
        --data_dir "/workspace/datasets/layout/unsv2_layout/${dataset}" \
        --coco_json_file "/workspace/datasets/layout/unsv2_layout/${dataset}/train.json" \
        --output_file "${output_dir}/${dataset_lower}_train.json" \
        --url_prefix "http://192.168.106.8/datasets/unsv2/layout/${dataset}/images" \
        --category_mapping_file "/workspace/youjiachen/workspace/ultralytics/scripts/data_process/layout_mapping_config/${dataset_lower}.yaml"
    
    # 处理验证集
    python label_studio_converter.py \
        --data_dir "/workspace/datasets/layout/unsv2_layout/${dataset}" \
        --coco_json_file "/workspace/datasets/layout/unsv2_layout/${dataset}/val.json" \
        --output_file "${output_dir}/${dataset_lower}_val.json" \
        --url_prefix "http://192.168.106.8/datasets/unsv2/layout/${dataset}/images" \
        --category_mapping_file "/workspace/youjiachen/workspace/ultralytics/scripts/data_process/layout_mapping_config/${dataset_lower}.yaml"
done

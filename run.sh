model_path=/new_data/yanghq/models/mistralai/Mixtral-8x7B-v0.1
model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite

gpuid=0,1,2,3,4,5,6
gpuid=0,1,

tasks=coqa
tasks=arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,winogrande
tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande
tasks=boolq,openbookqa,rte,winogrande
tasks=rte
tasks=openbookqa
tasks=rte,openbookqa

tasks=wikitext,gsm8k
# tasks=openbookqa
tasks=gsm8k

anno_list=(rou_cos_6_min rou_cos_6_max exp_cos_6_max rou_cos_4_min exp_cos_6_avg rou_cos_4_max exp_cos_6_min rou_cos_6_avg exp_cos_4_avg rou_cos_4_avg exp_cos_4_max exp_cos_4_min )
anno_list=(rou_mse_6_min rou_mse_6_max exp_mse_6_max rou_mse_4_min exp_mse_6_avg rou_mse_4_max exp_mse_6_min rou_mse_6_avg exp_mse_4_avg rou_mse_4_avg exp_mse_4_max exp_mse_4_min )


echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno
    
    CUDA_VISIBLE_DEVICES=$gpuid lm_eval --model hf \
        --tasks $tasks \
        --model_args pretrained=${model_path},parallelize=True,dtype=float16,max_memory_per_gpu=23100100100 \
        --trust_remote_code \
        --num_fewshot 0 \
        --output_path ./outputs/ \
        --batch_size 16 \
        # --limit 0.1 \
        # --merge ${anno} \
        # --use_cache ./cache/ \
        # --max_batch_size 128 \

done

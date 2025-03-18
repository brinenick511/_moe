model_path=${HOME}/models/mistralai/Mixtral-8x7B-v0.1
# model_path=${HOME}/models/deepseek-ai/DeepSeek-V2-Lite
# model_path=${HOME}/models/deepseek-ai/DeepSeek-V2-Lite-Chat
# model_path=${HOME}/models/Qwen/Qwen2.5-0.5B-Instruct

gpuid=0,1,2,3,4,5,6
gpuid=0,1

tasks=hellaswag,mmlu
tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande,gsm8k
tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande
# tasks=mmlu

NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=$gpuid lm_eval --model hf \
    --tasks $tasks \
    --model_args pretrained=${model_path},parallelize=True,dtype=bfloat16,max_memory_per_gpu=23100100100,trust_remote_code=True \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path ./outputs/ \
    --batch_size auto:2 \
    # --use_cache ./cache/ \
    # --limit 4 \
    # --merge ${anno} \
    # --max_batch_size 128 \


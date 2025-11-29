# CUDA_VISIBLE_DEVICES=0 python generate_act_scale_shift.py --model /root/dataset/Llama-2-7b

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log_rank/Llama-2-7b-w4a4_rank1 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75 --rotate_rank 1 --lr_plus

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log_rank/Llama-2-7b-w4a4_rank4 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75 --rotate_rank 4 --lr_plus

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log_rank/Llama-2-7b-w4a4_rank8 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75 --rotate_rank 8 --lr_plus

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log_rank/Llama-2-7b-w4a4_rank16 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75 --rotate_rank 16 --lr_plus

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log_rank/Llama-2-7b-w4a4_rank32 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75 --rotate_rank 32 --lr_plus

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log_rank/Llama-2-7b-w4a4_rank64 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75 --rotate_rank 64 --lr_plus
import os
import torch

from optimum.rbln import RBLNAutoModelForCausalLM


def main():
    model_id = "Qwen/Qwen3-0.6B"

    # Compile and export
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,                        # To compile the model, this argument must be True
        rbln_batch_size=8,                  # Maximum batch size for prefill
        rbln_max_seq_len=40_960,            # default "max_position_embeddings"
        rbln_tensor_parallel_size=1,        # Tensor parallelism
        rbln_attn_impl="flash_attn",
        rbln_kvcache_partition_len=8_192,   # Length of KV cache partitions for flash attention
        rbln_decoder_batch_sizes=[8, 4, 1], # Compile decoders for batch sizes 8, 4, and 1
        torch_dtype=torch.float16
    )

    # Save compiled results to disk
    model.save_pretrained("Qwen3-0.6B-fp16-dynamic")


if __name__ == "__main__":
    main()

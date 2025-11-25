import os
os.environ["VLLM_USE_V1"] = "0"
is_v1 = os.environ.get("VLLM_USE_V1", "1") != "0"
print("vLLM v1 enabled:", is_v1)
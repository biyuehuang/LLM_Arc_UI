test benchmark like 32in/32out and 1024in/128out

如果有两张卡，需要测试多batch size，或多instance的情况

```
$ sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.6.0.22_223734]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Xeon(R) Gold 5420+ 3.0 [2023.16.6.0.22_223734]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```

（1）1 instance per A770：ZE_AFFINITY_MASK=0 python3 chatglm2-chat.py &ZE_AFFINITY_MASK=1 python3 chatglm2-chat.py （每张卡分别加载一个模型，模型不共享）

（2）2 instance First A770：ZE_AFFINITY_MASK=0 python3 chatglm2-chat.py &ZE_AFFINITY_MASK=0 python3 chatglm2-chat.py  （模型不会共享，相当于一张卡加载2个模型）

（3）2 batch First A770：（模型共享，一张卡加载1个模型，推理2条等长的prompt）

```
batch_input = ["string input 1", "string input 2"]
input_ids = tokenizer(batch_input, return_tensors="pt").to(device)
output = llama_model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)
output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
```

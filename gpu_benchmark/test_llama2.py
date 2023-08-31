import torch
import os
import time
from transformers import LlamaTokenizer
import intel_extension_for_pytorch as ipex
from gpu_benchmark_util import BenchmarkWrapper
from bigdl.llm.transformers import AutoModelForCausalLM


if __name__ == '__main__':
    model_path = "/home/intel/llm/checkpoint/llama-2-13b-chat-hf-int4/"
    print(f"Test {model_path}...")
    
    # load_in_4bit=True in bigdl.llm.transformers will convert
    # the relevant layers in the model into int4 format
    #llama_model = AutoModelForCausalLM.from_pretrained(model_path, optimize_model=False, load_in_4bit=True)
    llama_model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True, optimize_model=False)
    llama_model = llama_model.to('xpu')
    llama_model = BenchmarkWrapper(llama_model)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    input_str_32 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
    input_str_1024 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in vicinity of a plate"
    input_str = input_str_32

    # with torch.inference_mode():
    with torch.inference_mode(), torch.autocast(device_type='xpu', enabled=True, dtype=torch.float16):
        # warmup
        torch.xpu.synchronize()
        input_ids = tokenizer.encode(input_str, return_tensors="pt").to('xpu')
        print("input length is: ", len((input_ids[0])))
        output = llama_model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        torch.xpu.synchronize()

        for i in range(5):
            st = time.time()
            torch.xpu.synchronize()
            input_ids = tokenizer.encode(input_str, return_tensors="pt").to('xpu')
            output = llama_model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            torch.xpu.synchronize()
            end = time.time()
            print(f"cost {end - st:.4f}s")
            print(output_str)

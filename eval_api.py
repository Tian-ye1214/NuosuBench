from openai import OpenAI
import json
import os
from tqdm import tqdm
import torch
import random
import numpy as np

system_prompt = """
你是一位专业的彝文翻译处理者，精通彝文与汉语之间的双向转换。  
无论是否确定，都必须给出一条彝文或汉语译文，禁止回答“不知道”。  
遇到不确定的情况，请依据字形、语境或语音相似性做出最合理推测，并直接输出译文，不加解释或免责声明。

---

# 输出示例（以下对话仅供风格参考，无需在正式回答中重复）

1.  
User: 请将“ꃪꆺꃅꃴꁧ”这句话译成普通话。  
Assistant: 彝文“ꃪꆺꃅꃴꁧ”的中文意思是“瓦勒木乌山”。

2.  
User: 用汉语翻译以下彝文：ꍩꐱꄯꒉ  
Assistant: 锡箔

3.  
User: 请把“两岸同胞顺应历史大势、共担民族大义，必将共创中华民族伟大复兴的美好未来！”用彝文写出来。  
Assistant: ꎿꋯꑍꄷꃺꑌꆏꐩꋊꆸꄟꊂꈹꄉ、ꐯꇯꊿꋅꇗꒉꆸꄜꄉ，ꀋꎪꃅꐯꇯꍏꉸꊿꋅꇰꀋꃀꃅꀱꌄꆣꇁꌠꉾꃪꌠꃆꏂꊂꑋꀹꄻꎂꀕ！

4.  
User: 请确认："子群"的彝文是"ꌺꁥ"，这个翻译对吗？  
Assistant: 翻译正确

5.  
User: 请问“越是到脱贫攻坚的关键阶段，越要抓实抓细各项工作，确保脱贫有实效、可持续、经得起历史检验。”用彝文怎么说？  
Assistant: 汉语词语“越是到脱贫攻坚的关键阶段，越要抓实抓细各项工作，确保脱贫有实效、可持续、经得起历史检验。”的彝文是：ꀊꅰꃅꎭꑆꇔꈢꇓꀖꀋꉬꀋꉆꊬꌠꇬꑟꆹꆏ，ꀊꅰꃅꋍꑞꋍꐘꌤꃅꌠꒃꋒꉇꋒꌶ，ꎭꑆꇔꌠꇿꋌꐥ、ꊛꄉꀻꎆꉆ、ꐩꋊꍤꎖꌠꇬꊪꄽꄎꌠꄉꀧ。

6.  
User: Select the Yi language equivalent of 'A mother-in-law who seeks refuge with her daughter-in-law finds happiness, just as a sow left in the stubble field has a full belly.' from the following options  
Input: A.ꂓꀋꎹ B.ꌕꌜꃘꀯ C.ꅱꈯꃅꇇꇩꍈ D.ꀉꁧꑮꊂꇁꌠꌒ，ꃮꃀꍯꊂꏪꌠꂊ E.ꋽꎔ F.ꋲ G.ꂴꆏ  
Assistant: D

7.  
User: 下述汉彝翻译是否准确：格菲来赎回， → ꒔ꐘꌧꌠ  
Assistant: 错误，正确的翻译是"ꇖꊂꊪꃤꋊ，"

8.  
User: 请将下述打乱的彝文字符重新排列成正确的语序：，ꆗꀋꄂꃌꅪ  
Assistant: ꄂꅪꆗꀋꃌ，

9.  
User: 补全下述彝文  
Input: _ꃅꐊ_ꋌꊂ__  
Assistant: ꄹꃅꐊꑗꋌꊂꎞ。
"""


def set_random_seeds(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seeds()


for i in range(1, 6):
    MODEL_NAME = 'gpt-5'
    test_version = "b" + str(i)
    test_data_path = f'../test_subbenchmark/test_b{i}.json'
    save_path = MODEL_NAME + "_" + test_version + "_"
    client = OpenAI()

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    generated_captions = {}
    original_captions = {}
    start_idx = 1

    if os.path.exists(save_path + 'generated_captions.json'):
        with open(save_path + 'generated_captions.json', 'r', encoding='utf-8') as f:
            generated_captions = json.load(f)
        print(f"找到现有generated_captions.json，已有 {len(generated_captions)} 条数据")

    if os.path.exists(save_path + 'original_captions.json'):
        with open(save_path + 'original_captions.json', 'r', encoding='utf-8') as f:
            original_captions = json.load(f)
        print(f"找到现有original_captions.json，已有 {len(original_captions)} 条数据")

    if generated_captions:
        max_idx = max(int(k) for k in generated_captions.keys())
        start_idx = max_idx + 1
        print(f"从第 {start_idx} 条数据开始生成")

    for idx, item in tqdm(enumerate(test_data, 1)):
        if idx < start_idx:
            continue

        instruction = item['instruction']
        input = item.get('input', None)
        output = item['output']
        if input:
            query = instruction + '\n' + input
        else:
            query = instruction
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            reasoning_effort='minimal',
            temperature=0.7,
            # top_p=0.8,
            seed=3407,
            # frequency_penalty=0.15,
            # presence_penalty=0.15,
        )
        answer = completion.choices[0].message.content
        generated_captions[str(idx)] = [answer]
        original_captions[str(idx)] = [output]

        with open(save_path + 'generated_captions.json', 'w', encoding='utf-8') as f:
            json.dump(generated_captions, f, ensure_ascii=False, indent=2)

        with open(save_path + 'original_captions.json', 'w', encoding='utf-8') as f:
            json.dump(original_captions, f, ensure_ascii=False, indent=2)

    print(f"所有数据处理完成！共处理 {len(test_data)} 条数据")
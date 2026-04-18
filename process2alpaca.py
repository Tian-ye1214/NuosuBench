import json
import random
from tqdm import tqdm
import multiprocessing as mp

system_prompt = """
您是一位专业的双语翻译人员，专精于中彝和彝中翻译。您的任务是提供清晰准确的翻译，同时确保：
1. 首要任务是保持译文的**准确性**，不丢失原文含义或避免歧义。
2. 尽可能确保译文在目标语言中**自然**且**优雅**，但前提是确保准确性。
3. 有时，您可能需要执行其他任务，例如：
- 根据提供的文本回答问题或解决问题。
- 必要时提供文化解释或背景信息，尤其是在文本涉及目标语言受众可能不熟悉的文化指涉或习语表达时。
4. 保持**语境一致性**，尤其是在存在多个相关任务的情况下。如果一项任务不仅仅需要翻译（例如，解决问题或解释概念），请在保持翻译完整性的同时做出相应的回应。
"""


def translation(data, position=0) -> list:
    result = []

    for item in tqdm(data, desc="翻译任务", position=position, leave=True):
        yi = item['yi']
        chinese = item['chinese']
        english = item.get('english', None)
        content = item.get('content', None)
        english_content = item.get('english_content', None)

        chinese2yi_instructions = [
            f"将“{chinese}”翻译成彝文。",
            f"请将下述内容译为彝文：{chinese}。",
            f"请问“{chinese}”用彝文怎么说？",
            f"告诉我“{chinese}”的彝文吗？",
            f"把“{chinese}”翻译成彝语。",
            f"请用彝文表达以下汉语内容：“{chinese}”。",
            f"“{chinese}”对应的彝文是什么？",
            f"“{chinese}”的彝文是？",
            f"请把“{chinese}”用彝文写出来。",
            f"请翻成彝文：{chinese}",
            f"彝文翻译：“{chinese}”。",
            f"将“{chinese}”译成彝文。",
            f"你知道“{chinese}”的彝文是？",
            f"请帮我把“{chinese}”翻成彝语，谢谢。",
            f"用彝文说“{chinese}”应该怎么说？",
            f"“{chinese}”的彝文翻译是？",
            f"如何用彝文表达“{chinese}”？",
            f"请用彝文翻译下面这句话：“{chinese}”。",
            f"“{chinese}”这句话用彝语怎么说？",
            f"请提供“{chinese}”的彝文对应说法。",
        ]

        yi2chinese_instructions = [
            f"请将彝文“{yi}”翻译为汉语。",
            f"请将下述彝文内容翻译成中文：“{yi}”。",
            f"“{yi}”用汉语怎么说？",
            f"“{yi}”的意思是？",
            f"请问“{yi}”翻成中文是什么？",
            f"把“{yi}”翻译成中文。",
            f"请用中文解释“{yi}”的意思。",
            f"“{yi}”在汉语中是什么意思？",
            f"请帮我把彝文“{yi}”译为汉语。",
            f"这段彝文“{yi}”用汉语如何表达？",
            f"请把“{yi}”翻译成汉语，谢谢。",
            f"“{yi}”的中文翻译是什么？",
            f"如何将“{yi}”这段彝文转写为中文？",
            f"用汉语翻译以下彝文：{yi}",
            f"请将彝语“{yi}”的意思用中文写出来。",
            f"我想知道“{yi}”用中文怎么说。",
            f"帮我翻译一下这段彝文：“{yi}”。",
            f"请将“{yi}”这句话译成普通话。",
            f"彝文“{yi}”在中文中怎么说？",
        ]

        chinese2yi_output = [
            f"{yi}",
            f"汉语词语“{chinese}”的彝文是：{yi}",
            f"彝文是“{yi}”。",
            f"是“{yi}”。",
            f"彝文的表达是“{yi}”。",
        ]

        yi2chinese_output = [
            f"{chinese}",
            f"彝文“{yi}”的中文意思是“{chinese}”",
            f"翻译成汉语是“{chinese}”",
        ]

        result.append({
            "instruction": random.choice(chinese2yi_instructions),
            "input": '',
            "output": random.choice(chinese2yi_output),
        })

        result.append({
            "instruction": random.choice(yi2chinese_instructions),
            "input": '',
            "output": random.choice(yi2chinese_output),
        })

        if english:
            english2yi_instructions = [
                f"请将\"{english}\"翻译成彝文。",
                f"请将下述内容译为彝文：\"{english}\"。",
                f"Express \"{english}\" in Yi language.",
                f"How do you say \"{english}\" in Yi language?",
                f"请把英文“{english}”翻译为彝语。",
                f"请用彝文写出“{english}”。",
                f"请问“{english}”用彝文怎么说？",
                f"“{english}”的彝语是什么？",
                f"你能告诉我“{english}”用彝文怎么表达吗？",
                f"Could you translate \"{english}\" into Yi language?",
                f"What is the Yi translation of \"{english}\"?",
                f"Translate the following English sentence into Yi: \"{english}\"",
                f"How would you write \"{english}\" in Yi?",
                f"Please provide the Yi equivalent of \"{english}\".",
            ]

            yi2english_instructions = [
                f"将“{yi}”译成标准英语表达。",
                f"“{yi}”的英文翻译是什么？",
                f"Translate \"{yi}\" into standard English expression.",
                f"How do you say \"{yi}\" in English?",
                f"请把彝文“{yi}”翻译成英文。",
                f"请用英文解释“{yi}”的含义。",
                f"“{yi}”在英语中怎么说？",
                f"请将以下彝语内容翻译为英文：{yi}",
                f"你能把“{yi}”用英语表达出来吗？",
                f"用英语写出“{yi}”的意思。",
                f"What does \"{yi}\" mean in English?",
                f"Please translate the Yi phrase \"{yi}\" into English.",
                f"Could you convert \"{yi}\" into English?",
                f"Give the English equivalent of \"{yi}\".",
                f"What is the standard English translation for \"{yi}\"?",
                f"How would you interpret \"{yi}\" in English?",
            ]

            english2yi_output = [
                f"{yi}",
                f"英语'{english}'的彝文是：{yi}",
                f"在彝文中，'{english}'写作'{yi}'",
                f"The Yi language for {english} is {yi}.",
                f"{yi} is the Yi language form of {english}.",
                f"彝文中写作“{yi}”。",
                f"“{yi}”是对应的彝文。",
            ]

            yi2english_output = [
                f"{english}.",
                f"彝文“{yi}”的英文是\"{english}\"，对应的汉语是“{chinese}”。",
                f"“{yi}”翻译成英语是\"{english}\"，“{chinese}”是对应的汉语。",
                f"'{yi}' corresponds to '{english}' in English and '{chinese}' in Chinese.",
                f"'{yi}' means '{english}', its Chinese is '{chinese}'。",
                f"英语翻译为\"{english}\"。",
            ]

            result.append({
                "instruction": random.choice(yi2english_instructions),
                "input": '',
                "output": random.choice(yi2english_output)
            })

            result.append({
                "instruction": random.choice(english2yi_instructions),
                "input": '',
                "output": random.choice(english2yi_output)
            })

        if content:
            chinese2yi_plus_instructions = [
                f"请将汉语“{chinese}”翻译成彝文。并解释其含义。",
                f"请将下述内容译为彝文然后介绍一下：{chinese}。",
                f"把“{chinese}”翻译成彝语说明对应文化。",
                f"“{chinese}”对应的彝文是什么？解释一下。",
                f"请把“{chinese}”用彝文写出来并解释。",
                f"彝语里怎么说“{chinese}”？是什么意思？",
                f"“{chinese}”的彝文翻译是？介绍解释一下。",
            ]

            yi2chinese_plus_instructions = [
                f"请将彝文“{yi}”翻译为汉语。并解释其含义。",
                f"请将下述彝文内容翻译成中文再介绍：“{yi}”。",
                f"“{yi}”用汉语怎么说？啥意思？",
                f"请问“{yi}”翻成中文是什么意思？",
                f"把“{yi}”翻译成中文并介绍介绍。",
                f"请用中文解释“{yi}”的意思与文化。",
                f"“{yi}”在汉语中是什么意思？如何解读？",
                f"请帮我把彝文“{yi}”译为汉语。再用汉语解释一下",
                f"这段彝文“{yi}”用汉语如何表达？啥意思？",
                f"请把“{yi}”翻译成汉语再介绍一下，谢谢。",
            ]

            chinese2yi_plus_output = [
                f"{yi}\n{content}",
                f"彝文是：{yi}\n{content}",
                f"在彝文中，“{chinese}”写作“{yi}”\n{content}",
                f"“{chinese}”的彝文是“{yi}”。\n{content}",
                f"“{yi}”是彝文写法。\n{content}",
            ]

            yi2chinese_plus_output = [
                f"{chinese}\n{content}",
                f"“{yi}”的中文意思是“{chinese}”\n{content}",
                f"“{yi}”翻译成汉语是“{chinese}”\n{content}",
                f"汉语为“{chinese}”。\n{content}",
                f"意为“{chinese}”。\n{content}",
            ]

            result.append({
                "instruction": random.choice(chinese2yi_plus_instructions),
                "input": '',
                "output": random.choice(chinese2yi_plus_output)
            })

            result.append({
                "instruction": random.choice(yi2chinese_plus_instructions),
                "input": '',
                "output": random.choice(yi2chinese_plus_output)
            })

        if english_content:
            english2yi_plus2_instructions = [
                f"请将“{english}”翻译成彝文，并说明其具体含义。",
                f"请用彝文准确表达“{english}”，并简要解读。",
                f"请把“{english}”翻译成彝文，同时介绍一下它的背景或含义。",
                f"请翻译“{english}”为彝语，并结合文化背景解释。",
                f"请写出“{english}”的彝文说法，并简明说明。",
                f"“{english}”用彝文怎么说？请结合含义解释清楚。",
                f"请用彝文表达“{english}”，并解读其意思。",
                f"请将“{english}”的彝语翻译和含义一起告诉我。",
                f"请把“{english}”用彝语翻译出来，并说明它的意思或用途。",
                f"Translate \"{english}\" into Yi language with a brief explanation.",
                f"Express \"{english}\" in Yi and elaborate on its meaning.",
                f"How would you translate \"{english}\" into Yi and explain what it conveys?",
                f"Please provide the Yi version of \"{english}\" along with its interpretation.",
                f"Could you translate \"{english}\" into Yi and give some context or explanation?",
                f"What’s the Yi equivalent of \"{english}\"? Please interpret its meaning.",
                f"Translate \"{english}\" into Yi and briefly explain its meaning or background.",
            ]

            yi2english_plus2_instructions = [
                f"请将“{yi}”翻译成标准英文，并解释其含义。",
                f"请用英语表达“{yi}”，顺便说明一下意思。",
                f"把“{yi}”翻译成英语，同时解读一下它的背景或用途。",
                f"请将彝语“{yi}”译成英文，并结合上下文解释。",
                f"请用英语准确表达“{yi}”，并简单介绍一下含义。",
                f"“{yi}”用英语怎么说？请顺便解释清楚。",
                f"请翻译“{yi}”为英文，并简要解读其意思。",
                f"请把“{yi}”的英文翻译和具体含义一起告诉我。",
                f"请将“{yi}”翻译成英语，附带说明它的含义或背景。",
                f"Translate \"{yi}\" into English with a brief explanation.",
                f"Express \"{yi}\" in English and elaborate on its meaning.",
                f"How would you translate \"{yi}\" into English and explain its significance?",
                f"Please provide the English version of \"{yi}\" along with its interpretation.",
                f"Could you translate \"{yi}\" into English and briefly explain the meaning?",
                f"What’s the English equivalent of \"{yi}\"? Please interpret its meaning clearly.",
                f"Translate \"{yi}\" into English and explain its background or cultural context.",
            ]

            english2yi_plus2_output = [
                f"{yi}\n{content}",
                f"{yi}\n{english_content}",
                f"英语'{english}'的彝文是：{yi}\n{content}",
                f"'{english}'写作'{yi}'\n{content}",
                f"The Yi language for {english} is {yi}.\n{english_content}",
                f"{yi} is the Yi language form of {english}.\n{english_content}",
                f"彝文为“{yi}”。\n{content}",
                f"“{yi}”是“对应的彝文。\n{content}",
            ]

            yi2english_plus2_output = [
                f"{english}\n{english_content}",
                f"“{yi}”的英文是\"{english}\"，对应的汉语是“{chinese}”\n{content}",
                f"“{yi}”翻译成英语是\"{english}\"。\n{content}",
                f"英文意为\"{english}\"。\n{content}",
            ]

            result.append({
                "instruction": random.choice(yi2english_plus2_instructions),
                "input": '',
                "output": random.choice(yi2english_plus2_output)
            })

            result.append({
                "instruction": random.choice(english2yi_plus2_instructions),
                "input": '',
                "output": random.choice(english2yi_plus2_output)
            })

    print(f'翻译共计{len(result)}条数据')
    return result


def choice(data, position=1) -> list:
    result = []

    for item in tqdm(data, desc="选择任务", position=position, leave=True):
        yi = item['yi']
        if len(yi) >= 30:
            continue
        chinese = item['chinese']
        english = item.get('english', None)
        num_options = random.randint(4, 8)

        yi_to_chinese_options = [chinese]
        other_items = [other for other in data if other != item]
        wrong_answers = random.sample(other_items, min(num_options - 1, len(other_items)))
        for wrong_item in wrong_answers:
            yi_to_chinese_options.append(wrong_item['chinese'])

        random.shuffle(yi_to_chinese_options)
        correct_index = yi_to_chinese_options.index(chinese)
        correct_letter = chr(ord('A') + correct_index)

        options_str = ""
        for i, option in enumerate(yi_to_chinese_options):
            letter = chr(ord('A') + i)
            options_str += f"{letter}.{option} "
        options_str = options_str.strip()

        result.append({
            "instruction": f"从下述多个选项中选择出“{yi}”对应的中文",
            "input": options_str,
            "output": correct_letter
        })

        chinese_to_yi_options = [yi]
        for wrong_item in wrong_answers:
            chinese_to_yi_options.append(wrong_item['yi'])

        random.shuffle(chinese_to_yi_options)
        correct_index = chinese_to_yi_options.index(yi)
        correct_letter = chr(ord('A') + correct_index)

        options_str = ""
        for i, option in enumerate(chinese_to_yi_options):
            letter = chr(ord('A') + i)
            options_str += f"{letter}.{option} "
        options_str = options_str.strip()

        result.append({
            "instruction": f"从下述多个选项中选择出“{chinese}”对应的彝文",
            "input": options_str,
            "output": correct_letter
        })

        if english:
            english_to_yi_options = [yi]
            english_items = [other for other in data if other != item and other.get('english')]
            if len(english_items) >= num_options - 1:
                english_wrong_answers = random.sample(english_items, num_options - 1)
                for wrong_item in english_wrong_answers:
                    english_to_yi_options.append(wrong_item['yi'])

                random.shuffle(english_to_yi_options)
                correct_index = english_to_yi_options.index(yi)
                correct_letter = chr(ord('A') + correct_index)

                options_str = ""
                for i, option in enumerate(english_to_yi_options):
                    letter = chr(ord('A') + i)
                    options_str += f"{letter}.{option} "
                options_str = options_str.strip()

                result.append({
                    "instruction": f"Select the Yi language equivalent of '{english}' from the following options",
                    "input": options_str,
                    "output": correct_letter
                })

            yi_to_english_options = [english]
            if len(english_items) >= num_options - 1:
                for wrong_item in english_wrong_answers:
                    yi_to_english_options.append(wrong_item['english'])

                random.shuffle(yi_to_english_options)
                correct_index = yi_to_english_options.index(english)
                correct_letter = chr(ord('A') + correct_index)
                options_str = ""
                for i, option in enumerate(yi_to_english_options):
                    letter = chr(ord('A') + i)
                    options_str += f"{letter}.{option} "
                options_str = options_str.strip()

                result.append({
                    "instruction": f"Select the English corresponding to '{yi}' from the following options",
                    "input": options_str,
                    "output": correct_letter
                })

    print(f'选择共计{len(result)}条数据')
    return result


def cloze(data, position=2) -> list:
    result = []

    for item in tqdm(data, desc="完形填空任务", position=position, leave=True):
        yi = item['yi']
        chinese = item['chinese']

        if len(yi) <= 15:
            continue

        min_mask = max(1, int(len(yi) * 0.1))
        max_mask = min(len(yi) - 1, int(len(yi) * 0.5))
        num_to_mask = random.randint(min_mask, max_mask)
        mask_positions = random.sample(range(len(yi)), num_to_mask)

        masked_yi = list(yi)
        for pos in mask_positions:
            masked_yi[pos] = '_'
        masked_text = ''.join(masked_yi)

        cloze_instructions = [
            "补全下述彝文",
            "请将下面彝文中的“_”补全",
            f"补全下述意思为“{chinese}”的彝文",
        ]

        result.append({
            "instruction": random.choice(cloze_instructions),
            "input": masked_text,
            "output": yi
        })

    print(f'完形填空共计{len(result)}条数据')
    return result


def correct(data, position=3) -> list:
    result = []

    for item in tqdm(data, desc="纠错任务", position=position, leave=True):
        yi = item['yi']
        chinese = item['chinese']
        correct_instructions = [
            f"下述意为“{chinese}”彝文中可能出现错误，请改正为正确的表达",
            f"请检查下述彝文是否有错误，如有错误请改正",
            f"表示“{chinese}”的彝文可能有错误，请检查并改正",
            f"下述彝文是否对应“{chinese}”？",
            f"下述彝文表达是否正确？如有错误请改正",
            f"下面这段意味“{chinese}”彝文有什么问题吗？",
        ]

        if random.random() < 0.8 and len(yi) > 15:
            corrupted_yi = list(yi)
            error_type = random.choice(['replace', 'delete', 'insert', 'swap'])

            if error_type == 'replace' and len(corrupted_yi) > 0:
                pos = random.randint(0, len(corrupted_yi) - 1)
                all_yi_chars = set()
                for other_item in data:
                    all_yi_chars.update(other_item['yi'])
                all_yi_chars = list(all_yi_chars)
                if all_yi_chars:
                    new_char = random.choice(all_yi_chars)
                    corrupted_yi[pos] = new_char

            elif error_type == 'delete' and len(corrupted_yi) > 1:
                pos = random.randint(0, len(corrupted_yi) - 1)
                corrupted_yi.pop(pos)

            elif error_type == 'insert':
                pos = random.randint(0, len(corrupted_yi))
                all_yi_chars = set()
                for other_item in data:
                    all_yi_chars.update(other_item['yi'])
                all_yi_chars = list(all_yi_chars)
                if all_yi_chars:
                    new_char = random.choice(all_yi_chars)
                    corrupted_yi.insert(pos, new_char)

            elif error_type == 'swap' and len(corrupted_yi) > 1:
                pos = random.randint(0, len(corrupted_yi) - 2)
                corrupted_yi[pos], corrupted_yi[pos + 1] = corrupted_yi[pos + 1], corrupted_yi[pos]

            corrupted_text = ''.join(corrupted_yi)

            result.append({
                "instruction": random.choice(correct_instructions),
                "input": corrupted_text,
                "output": yi
            })
        else:
            result.append({
                "instruction": random.choice(correct_instructions),
                "input": yi,
                "output": f"没有错误。"
            })

    print(f'改错题共计{len(result)}条数据')
    return result


def judge(data, position=4) -> list:
    result = []

    for item in tqdm(data, desc="判断任务", position=position, leave=True):
        yi = item['yi']
        chinese = item['chinese']
        english = item.get('english', None)

        correct_judge_instructions = [
            f"判断下述翻译是否正确：彝文\"{yi}\"对应汉语\"{chinese}\"",
            f"请判断：\"{yi}\"翻译成\"{chinese}\"是否正确？",
            f"下述彝汉翻译是否准确：{yi} → {chinese}",
            f"验证翻译正确性：彝文\"{yi}\"是否等于汉语\"{chinese}\"？",
            f"请确认：\"{chinese}\"的彝文是\"{yi}\"，这个翻译对吗？",
        ]

        correct_outputs = [
            "正确",
            "翻译正确",
            "是的，翻译正确",
            "正确，彝文和汉语对应准确",
        ]

        result.append({
            "instruction": random.choice(correct_judge_instructions),
            "input": '',
            "output": random.choice(correct_outputs)
        })

        other_items = [other for other in data if other != item]
        if other_items:
            wrong_item = random.choice(other_items)
            wrong_chinese = wrong_item['chinese']

            wrong_judge_instructions = [
                f"判断下述翻译是否正确：彝文\"{yi}\"对应汉语\"{wrong_chinese}\"",
                f"请判断：\"{yi}\"翻译成\"{wrong_chinese}\"是否正确？",
                f"下述彝汉翻译是否准确：{yi} → {wrong_chinese}",
                f"验证翻译正确性：彝文\"{yi}\"是否等于汉语\"{wrong_chinese}\"？",
            ]

            wrong_outputs = [
                f"错误，正确翻译应该是：{chinese}",
                f"翻译错误，\"{yi}\"的正确汉语是\"{chinese}\"",
                f"不正确，彝文\"{yi}\"对应的汉语是\"{chinese}\"",
                f"错误，正确的翻译是\"{chinese}\"",
            ]

            result.append({
                "instruction": random.choice(wrong_judge_instructions),
                "input": '',
                "output": random.choice(wrong_outputs)
            })

        # 汉语到彝文的正确判断
        correct_chinese_to_yi_instructions = [
            f"判断下述翻译是否正确：汉语\"{chinese}\"对应彝文\"{yi}\"",
            f"请判断：\"{chinese}\"翻译成\"{yi}\"是否正确？",
            f"下述汉彝翻译是否准确：{chinese} → {yi}",
            f"验证翻译正确性：汉语\"{chinese}\"是否等于彝文\"{yi}\"？",
            f"请确认：\"{chinese}\"的彝文是\"{yi}\"，这个翻译对吗？",
        ]

        result.append({
            "instruction": random.choice(correct_chinese_to_yi_instructions),
            "input": '',
            "output": random.choice(correct_outputs)
        })

        # 汉语到彝文的错误判断
        if other_items:
            wrong_yi_item = random.choice(other_items)
            wrong_yi = wrong_yi_item['yi']

            wrong_chinese_to_yi_instructions = [
                f"判断下述翻译是否正确：汉语\"{chinese}\"对应彝文\"{wrong_yi}\"",
                f"请判断：\"{chinese}\"翻译成\"{wrong_yi}\"是否正确？",
                f"下述汉彝翻译是否准确：{chinese} → {wrong_yi}",
                f"验证翻译正确性：汉语\"{chinese}\"是否等于彝文\"{wrong_yi}\"？",
            ]

            wrong_chinese_to_yi_outputs = [
                f"错误，正确翻译应该是：{yi}",
                f"翻译错误，\"{chinese}\"的正确彝文是\"{yi}\"",
                f"不正确，汉语\"{chinese}\"对应的彝文是\"{yi}\"",
                f"错误，正确的翻译是\"{yi}\"",
            ]

            result.append({
                "instruction": random.choice(wrong_chinese_to_yi_instructions),
                "input": '',
                "output": random.choice(wrong_chinese_to_yi_outputs)
            })

        if english:
            result.append({
                "instruction": f"Judge if this translation is correct: '{yi}' means '{english}' in English",
                "input": '',
                "output": "Correct"
            })

            # 错误的英文判断
            if other_items:
                english_items = [other for other in other_items if other.get('english')]
                if english_items:
                    wrong_english_item = random.choice(english_items)
                    wrong_english = wrong_english_item['english']

                    result.append({
                        "instruction": f"Judge if this translation is correct: '{yi}' means '{wrong_english}' in English",
                        "input": '',
                        "output": f"Incorrect, '{yi}' means '{english}' in English"
                    })

    print(f'判断题共计{len(result)}条数据')
    return result


def reorder(data, position=5) -> list:
    result = []

    for item in tqdm(data, desc="语序恢复任务", position=position, leave=True):
        yi = item['yi']
        chinese = item['chinese']

        if len(yi) <= 15:
            continue

        yi_chars = list(yi)
        random.shuffle(yi_chars)
        shuffled_yi = ''.join(yi_chars)

        # 生成指令变体
        reorder_instructions = [
            f"请将下述打乱的彝文字符重新排列成正确的语序：{shuffled_yi}",
            f"以下彝文字符顺序被打乱了，请恢复正确的排列：{shuffled_yi}",
            f"请将下述打乱的彝文字符重新排列，使其表达\"{chinese}\"的含义：{shuffled_yi}",
            f"已知含义为\"{chinese}\"，请将打乱的彝文\"{shuffled_yi}\"恢复正确语序",
            f"这些彝文字符的意思是\"{chinese}\"，请重新排列：{shuffled_yi}",
        ]
        outputs = [
            f"正确的彝文是：{yi}",
            f"恢复后的正确语序为：{yi}",
            f"{yi}",
        ]

        result.append({
            "instruction": random.choice(reorder_instructions),
            "input": '',
            "output": random.choice(outputs)
        })

    print(f'语序恢复题共计{len(result)}条数据')
    return result


def execute_function_with_data(func_info):
    """执行单个函数的包装器"""
    func, data, position = func_info
    return func(data, position)


if __name__ == "__main__":
    input_file = "./Yi-Json.json"
    output_file = "./Yi-1.0.1-Alpaca.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions_with_params = [
        (translation, data, 0),
        (choice, data, 1),
        (cloze, data, 2),
        (correct, data, 3),
        (judge, data, 4),
        (reorder, data, 5)
    ]

    with mp.Pool(processes=6) as pool:
        results = pool.map(execute_function_with_data, functions_with_params)

    all_results = []
    for result in results:
        all_results.extend(result)

    print(f"总共生成 {len(all_results)} 条数据")
    random.shuffle(all_results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到 {output_file}")

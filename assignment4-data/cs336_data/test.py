import regex

# 我们要测试的文本
text_to_test = "(283)-182-3829"

# 你的正则表达式
# 我把 | 两边的空格去掉了，并加上了 \K
pat1=r'\d{3}[-]\d{3}[-]\d{4}'
pat2=r'\(\d{3}\)[-]\d{3}[-]\d{4}'
pat3=r'(?:\(\d{3}\)|\d{3})[-]\d{3}[-]\d{4}'
pat4=r'(?:\(\d{3}\)|\d{3})[-\s\.]*\d{3}[-\s\.]*\d{4}'
pat5=r'(?:\(\d{3}\)|\d{3})[-\s\.]*\d{3}[-\s\.]*\d{4}\b'
pat6=r'\b(?:\+?1[-\s\.]*)?(?:\(\d{3}\)|\d{3})[-\s\.]*\d{3}[-\s\.]*\d{4}\b'
patlist = [pat1, pat2, pat3, pat4, pat5, pat6]


# 编译
for i, pat_str in enumerate(patlist):
    pattern = regex.compile(pat_str)

    # --- 测试查找 ---
    print(f"\n使用正则表达式模式 {i+1}: '{pat_str}'")
    match = pattern.search(text_to_test)
    if match:
        print(f"测试查找: 成功！ 匹配到的内容是: '{match.group(0)}'")
    else:
        print("测试查找: 失败！ 没有找到匹配项。")

    # --- 测试替换 ---
    masked_text, num_subs = pattern.subn('|||PHONE_NUMBER|||', text_to_test)
    print(f"测试替换: 替换了 {num_subs} 次。 结果是: '{masked_text}'")
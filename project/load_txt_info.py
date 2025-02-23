# -*- coding: utf-8 -*-


from openai import OpenAI

# RedmiBook Pro 14
text_intro = text_tro = f'''
之前我们重复了无数次“2025年是笔记本电脑的大年”这句话，这是有理有据的。因为2025年不仅CPU、GPU会迭代，国补从开年就有，新模具也会更多，整个市场都会活跃不少。相比之下2024年就冷清许多了。

由于芯片整体迭代，性能提升的同时笔记本电脑的续航能力也会获得“整体进步”。那么续航提升会覆盖到哪些产品呢？今天我们就来简单分析一下：

小米 RedmiBook Pro 14
它的配置如下：
- Ultra 5 125H 处理器
- 32GB LPDDR5x 7467MHz 内存
- 1TB固态硬盘
- 14.0英寸 2880×1800分辨率 100%sRGB色域 120Hz刷新率IPS屏
- 电池容量 77Wh
- 厚度 16.7~17.8mm
- 重量 1.45kg
- 适配器重量 269g

参考售价：4766元

它的优缺点如下：
优点！
1. 续航表现较好
2. 适配器较为便携
3. 支持HyperOS，小米的多设备协同方便

缺点！
1. 内部扩展性一般，双M.2均为2242规格
2. 满载下风扇噪音较大
3. 屏幕有轻微油腻感

【升级建议】这台笔记本电脑拆机并不难，卸下D壳的螺丝后即可取下后盖。双通道32GB LPDDR5x 7467MHz内存能满足大部分用途的需求，内存为板载无法更换。测试机的固态硬盘容量为1TB，型号为长江存储 PC300，支持PCIe4.0×4和NVMe，机器还有一个2242规格的M.2插槽，如有需要可以自行加装固态硬盘。

【购买建议】
1. 对续航时长有一定要求
2. 需要小米设备互联
3. 对硬盘规格有一定了解

小米 RedmiBook Pro 14是一款即将迭代的笔记本，会从Ultra 100H迭代至Ultra 200H，属于笔记本中最主流的更新策略。（其他地方也可能有改变，具体等评测吧）

屏幕方面，实测色域容积105.3%sRGB，色域覆盖99.6%sRGB，以sRGB为参考，平均ΔE 0.97，最大ΔE 2.98，实测屏幕最大亮度约424nits。

接口方面，机身左侧依次为USB-C 10Gbps（支持DP）、Type-C、HDMI 2.1和3.5mm耳机孔；机身右侧为两个雷电4 USB-C端口。 

参考售价：4766元

【总体小结】
身为一款轻薄本，RedmiBook Pro 14搭载了77Wh电池，按照笔吧的标准，实测续航水平8小时56分钟，接近9小时的续航成绩，算是轻薄本中不错的水平。在2025年，中低价位段主要由“马甲处理器”占据，到了主流5000元以上的价位段，我们能看到Ultra 200H处理器实装在笔记本电脑上。根据我们初步的实测总结，搭载新款处理器的轻薄本、全能本，续航水平会有提升，在笔吧的续航脚本下，夸张的甚至能超过11小时。堪比Ultra 200V处理器，都快撵上苹果M芯片了。但是续航提升的好消息并不一定包括游戏本，游戏本芯片架构和台式机更接近，它们可不是为了省电而生的，所以大家不用报太多希望。

在未来，脚本实测8小时以上续航的电脑会有很多，即便是中高负载下使用，也能支撑5小时左右。我相信今年轻薄本的续航问题会得到彻底解决。
'''

additional_info = "The laptop has a network cable interface and a type-C interface.\nThe laptop has a network cable interface and a type-C interface.\nThe laptop's back cover is white.\nNo, there is no numeric keypad visible in the image. The keyboard shown is a standard QWERTY layout without a separate numeric keypad."
save_file_name = 'RedmiBook Pro 14'


merge_prompt = f'''
Here we have one introduction of one laptop, Production Introduction:{text_intro}, and we have some additional information related to the laptop, Additional Information:{additional_info}.
Now merge the Additional Information into the Production Introduction with following instructions:
1. Only merge the information in Additional Information that is related to the laptop which is introduced in Production Introduction.
2. Keep the format of the Production Introduction.
3. If the additional information is about the laptop's color or hardware condition or something familiar, add the additional information into the '配置' section.
4. If the additional information is about the laptop's performance druing running some applications, and there's no such a section in Production Introduction, create one new section '性能表现' below the section '拆机实拍', and put additional information there.If in addtional information, the performance test results include a description of the specific test settings and environment, include these settings in the final output.
5. If there's conflict between original Production Introduction and Additional Information,stick to the Production Introduction.
6. If the laptop doesn't have a nemeric keypad, clearly point out it in the '配置' section.
'''


def integrate_formation(merge_prompt, api_key="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"):
    client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
    completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":merge_prompt}],
  temperature=0,
  top_p=1,
  max_tokens=2048
)
    return completion.choices[0].message.content

print(integrate_formation(merge_prompt, api_key="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"))
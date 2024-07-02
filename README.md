
# AlPHTO 阿法兔 —— 谁是卧底

<br />
<!-- PROJECT LOGO -->
<p align="center">
  <a href="">
    <img src="assets/1.png" alt="Logo" width="60%">
  </a>

<h3 align="center">ALPHTO</h3>
  <p align="center">
    <br />
    <a href="https://openxlab.org.cn/apps/detail/HinGwenWong/Streamer-Sales">查看Demo</a>
    ·
    
  </p>
</p>


## 📢 介绍

**AlPHTO 阿法兔 —— AI智力游戏模型** 

阿法狗已为大家展示了，AI在围棋上的一统江湖，那么大语言模型LLM是否也可以做到了？

相比于大语言模型在工业上的应用，在AI游戏中更能体现出大语言模型的智慧。

模型采用用 [InternLM2](https://github.com/InternLM/InternLM) 和 [lmdeploy](https://github.com/InternLM/lmdeploy)部署。也兼容其他模型，及API形式。

**功能点总结：**

- 📜 大语言模型的反思和推理能力
- 🚀 AI在游戏中进行角色扮演
- 📚 AI支持为游戏生成内容
- 🎙️ 多agent共同参与游戏

**当前已经可以玩的游戏：**
- 🔊 猜谜语
- 🦸 五子棋
- 🌐 谁是卧底
- 🎉 海龟汤

 <a href="">
    <img src="assets/2.png" alt="alphato-game" width="100%">
  </a>

**文档最后有微信群，欢迎加入一起探讨更多的可能！** 🎉

**开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐⭐ , 您的 star ⭐是我们最大的鼓励，谢谢各位！**  

---

## 快速启动：
---
```bash
git clone https://github.com/BarryYin/AlphaTo

conda create -n alphato python=3.10
conda activate alphato
pip install -r requirements.txt

#### install

python app.py

#### 配置
默认大模型支持为 InternLM2 
如果要切换模型，需要在 model_configs.json  为你选择的模型添加API Key

```


## 项目说明：

#### 🧱 整体架构
 <a href="">
    <img src="assets/3.png" alt="alphato-game" width="60%">
  </a>


#### 🌐 游戏搭建思路
 <a href="">
    <img src="assets/4.png" alt="alphato-game" width="60%">
  </a>


#### 🦸 研究方向
- 多agent 支持的AI游戏
- AI的智力表现和自我反思能力
- 人类对AI参与游戏的感受



license: Apache License 2.0
---


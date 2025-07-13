<div align=center>
  <img src="imgs/QwT_illustration.png" width="500px" />
</div>

- transformers = 4.53.1

## 📌 Changelog

- **2025-07-08**: Release code implementing QwT + OPT, evaluated on WikiText. <img src="imgs/new.gif" alt="NEW" width="40"/>
- **2025-07-08**: Implementing QwT + Llama, under evaluated. 
- **2025-07-08**: QwT + OPT/Llama is good, implement per group compress. 
- **2025-07-08**: Per group compress under evaluated. 
- **2025-07-09**: Evaluate per group compress. 
- **2025-07-09**: Implementing QwT + Qwen, result not good, under optimization. 
- **2025-07-10**: Llama, Qwen2.5-0.5b, Qwen2.5-7b result is ok. 
- **2025-07-10**: Llama, Qwen2.5-0.5b, Qwen2.5-7b base quant clamp logged. 
- **2025-07-11**: Llama, Qwen2.5-0.5b, Qwen2.5-7b clamp-qwt good. 
- **2025-07-13**: Llama, Qwen2.5 clamp-qwt logged. 
- **2025-07-13**: Pack quantLinear. 
---

|             | Llama-1.1b | Llama-2-7b | Llama-3-8b | Qwen2.5-0.5b | Qwen2.5-0.5b | Qwen2.5-7b |
|:-----------:|:----------:|:----------:|:----------:|:------------:|:------------:|:----------:|
| Method      | PPL        | PPL        | PPL        | PPL          | PPL          | PPL        |
| bf16        | 7.973      | 5.473      | 6.137      | 13.076       | 9.2696       | 6.850      |
| W4A8        | 9.531      | 6.202      | 8.667      | 24.109       | 13.8900      | 9.295      |
| Clamp       | 15.162     | 8.498      | 16.054     | 37.500       | 24.5160      | 15.753     |
| Clamp + QwT | `9.1557`   | `6.2438`   | `8.7760`   | `19.8956`    | `13.4232`    | `9.4030`   |
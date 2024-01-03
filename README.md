### FateZero-UBW: Uniform-smoothing Binary masks per Word (IW F'23)

[Anlon Zhu](https://github.com/anlon-zhu)

## 🎏 Abstract
Diffusion-based text-to-video models have seen rapid growth, facilitating complex generation techniques such as generative text-to-video-editing (T2Ve). In particular, the Tune-A-Video text-to-video allows for one-shot tuning of input videos through DDIM inversion and generation of edited outputs via a denoising UNet; recent work in the FateZero has even enabled zero-shot editing through blended attention maps during the inversion stage. However, since these models are evaluated on easy-to-edit single-subject videos, the quality of generated videos significantly decreases on more diverse datasets––such as flickering and visual artifacts. Therefore, prompt engineering can be the primary barrier-to-entry for using these models. In this paper, I propose FateZero UBW, an extension to the FateZero model that 1) smooths the function for blended attention maps and 2) introduces novel hyperparameters to improve the overall configurability of the FateZero model. By introducing novel hyperparameters and refining the masking function, the model achieves improved fidelity to text prompts and reduced artifacts, which is especially striking in noisy video environments. Evaluation on diverse videos demonstrates reduced artifacts without sacrificing temporal coherence when compared to the original model, marking an advancement in the potential applications of T2Ve models.

## 💗 Acknowledgements

This repository builds off of [FateZero](https://fate-zero-edit.github.io/), which itself borrows heavily from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [prompt-to-prompt](https://github.com/google/prompt-to-prompt/). Thanks to the authors for sharing their code and models.

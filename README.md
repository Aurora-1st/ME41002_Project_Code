# DenseSTSR: Dense Connected Swin Transformer for Oral OCTA Image Super-resolution Reconstruction
## Architecture
![Figure 1: Framework of the proposed DenseSTSR network with 4 DDSTBs and 4 STBs.](/imgs/model_structure.png)
### Patch Extraction
* CNN - shallow features extraction

### Deep Feature Extraction
* Multi-head Self-Attention
* Dense connection
* Residual connection

### Post Upsampling
* PixelShuffle - upsampling
* CNN - reconstruction

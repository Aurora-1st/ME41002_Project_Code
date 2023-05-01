# DenseSTSR: Dense Connected Swin Transformer for Oral OCTA Image Super-resolution Reconstruction
## Architecture
![The architecture of DenseSTSR.](/imgs/model_structure.png)
### Patch Extraction
* CNN - shallow features extraction

### Deep Feature Extraction
* Multi-head Self-Attention
* Dense connection
* Residual connection

### Post Upsampling
* PixelShuffle - upsampling
* CNN - reconstruction

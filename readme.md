# RAG chatbot

This is my RAG project, which helps me read scientific papers by allowing me to ask questions about them. This setup matches resources I have available as I would've used different (stronger) models if I had better specs. 

----

Anyhow, make sure to install torch, torchvision & torchaudio + CUDA (which is compatable with your version of CUDA)

```console
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/<CUDA_VERSION>
```

Change <CUDA_VERSION> with your version of CUDA. If you don't know your version of CUDA, you can easily check it by:

```console
nvidia-smi
```

Some requirements are really specific and tied to the specific versions of other packages so that's why there are some restrictions on that.
import torch
print("CUDA disponible :", torch.cuda.is_available())
print("Nombre de GPUs disponibles :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))

from models.unet import UNet
def get_model(name):
    models = {"unet": UNet}
    if name.lower() not in models: raise ValueError(f"Model {name} not supported.")
    return models[name.lower()]()

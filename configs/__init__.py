from .config_img_processing import config_image_processing
from .config_dataset import config_dataset
from .config_model1_net import config_model1_net
from .config_model2_gan import config_model2_gan
# import copy

class ConfigDict(dict):
    """Dictionary subclass that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        # Convert nested dictionaries to ConfigDict objects
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

# Create a singleton config instance for direct import

config_img_proc = ConfigDict(config_image_processing)
config_dataset = ConfigDict(config_dataset)
config_model_net = ConfigDict(config_model1_net)
config_model_gan = ConfigDict(config_model2_gan)
# def load_config():
#     """Returns a fresh config that allows attribute-style access."""
#     return ConfigDict(copy.deepcopy(DEFAULT_CONFIG))
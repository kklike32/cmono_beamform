import yaml
import os

# have to do this so pytest can find settings.yml (tests still require settings.yml because individual SR components use it)
current = os.path.dirname(os.path.realpath(__file__))
settings_file = os.path.join(current, "settings.yml")

with open(settings_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

import os
import json
import UISettings # UI basic settings module


print(os.getcwd())

fileName = 'exp_config.json'

config_default = UISettings.get_config()

with open(fileName, 'w') as f:

    line = json.dumps(config_default, indent=4) + "\n"
    f.write(line)
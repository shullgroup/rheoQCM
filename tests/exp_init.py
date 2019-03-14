



from UISettings import settings_init 
import json

init = json.dumps(settings_init, indent=4) + '\n'

with open('settings_init.json', 'w') as f:
    f.write(init)

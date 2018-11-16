import json
from UISettings import settings_default

with open('settings_default.json', 'w') as f:
    line = json.dumps(settings_default, indent=4) + "\n"
    f.write(line)


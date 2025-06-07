import os
from gnng import TH_COMPILE
# Function to read the flag from environment variable
def get_torch_compile_trigger():
    return os.environ.get('TORCH_COMPILE_TRIGGER', TH_COMPILE) is True

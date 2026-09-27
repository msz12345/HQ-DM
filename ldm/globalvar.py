# Diffusion inputs are collected only when a calibration tool opts in. Keeping
# this disabled by default prevents ordinary sampling/training from retaining
# every denoising input in host memory.
diffusion_input_list = []
_input_collection_enabled = False


def enableInputCollection(enabled=True):
    global _input_collection_enabled
    _input_collection_enabled = bool(enabled)


def inputCollectionEnabled():
    return _input_collection_enabled


def clearInputList():
    diffusion_input_list.clear()

def appendInput(value):
    if _input_collection_enabled:
        diffusion_input_list.append(value)

def getInputList():
    return diffusion_input_list


global optimizer_state_list
optimizer_state_list = []
def init_state_list(num_steps):
    for _ in range(num_steps):
        optimizer_state_list.append([])

def saveStep(step, state):
    optimizer_state_list[step].append(state)

def getStep(step):
    if len(optimizer_state_list[step]) == 0:
        return None
    else:
        return optimizer_state_list[step].pop()

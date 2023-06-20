import os
import platform
import subprocess

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"])
        output = output.decode('utf-8')
        version_line = [line for line in output.split('\n') if 'release' in line][0]
        version = version_line.split(' ')[-1].replace(',', '')
        return version.replace('.', '')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def install_requirements():
    # determine the OS
    os_type = platform.system().lower()

    if os_type == "darwin":  # MacOS
        torch_command = ["pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/nightly/cpu"]
    elif os_type in ["linux", "windows"]:  # Linux or Windows
        cuda_version = get_cuda_version()
        if cuda_version is None and os_type == "windows":
            torch_command = ["pip", "install", "torch", "torchvision", "torchaudio"]
        elif cuda_version is None:
            torch_command = ["pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/nightly/cpu"]
        elif int(cuda_version) < 117:
            torch_command = ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/cu{cuda_version}"]
        elif int(cuda_version) == 117:
            torch_command = ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/cu{cuda_version}"]
        elif int(cuda_version) == 118:
            torch_command = ["pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/nightly/cu{cuda_version}"]
        else:
            torch_command = ["pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/nightly/cu{cuda_version}"]
    else:
        raise ValueError("Unsupported operating system: " + os_type)

    # install PyTorch
    subprocess.check_call(torch_command)
    
    # install other requirements
    requirements_command = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(requirements_command)

# Call the function
install_requirements()

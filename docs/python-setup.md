# Python Virtual Environments

Python handles its environment and packages using virtual environments. To run the python code in this repository, you will need to setup your own python virtual environment. This markdown file is meant to provide some instructions on how to manage a python virtual environment.

For more information on the topic, refer to the following link: https://docs.python.org/3/library/venv.html

### Disclaimer

For these instructions to work, you must already have python installed on your system.

The instructions in this file are dependent on the shells listed below. If you are using a different shell, these instructions may not work.

Again, for more information on the topic refer to the official python documentation.

## Setup

Note: Your system may use another command than `python` (e.g., `python3`). If this is the case, for this section specifically you may need to modify the commands to refer to the correct python command.

To create a python virtual environment, execute one of the following commands in your shell from the repository's root directory:

**Bash command (for POSIX systems)**
```bash
python -m venv ./.venv
```

**Powershell command (for Windows systems)**
```powershell
python -m venv .\.venv
```

Running this command will create a directory containing your python virtual environment. Note that, due to the .gitignore file, the new directory should not be committed to the repository.

## Activation

To use a python virtual environment, it must first be activated. This is done by running a script in the virtual environment's directory. The name/location of the script will vary depending on your platform.

To activate your python virtual environment, execute the following commands from the repository's root directory:

**Bash command (for POSIX systems)**
```bash
source ./.venv/bin/activate
```

**Powershell command (for Windows systems)**
```powershell
.\.venv\Scripts\Activate.ps1
```

Once activated, calls to commands like `python` or `pip` will use the instances of these commands stored in your virtual environment.

Your virtual environment should be active when executing python code in this repository.

## Deactivation

To deactivate your virtual environment, type the following into your shell:

**Bash command (for POSIX systems)**
```bash
deactivate
```

**Powershell command (for Windows systems)**
```powershell
deactivate
```

## Package installation

The python packages used in this repository are specified in the `requirements.txt` file in this repository.

To install or update the packages for your virtual environment, execute the following with an activated environment:

**Bash command (for POSIX systems)**
```bash
pip install -r requirements.txt
```

**Powershell command (for Windows systems)**
```powershell
pip install -r requirements.txt
```

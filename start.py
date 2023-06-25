import subprocess
import os


# ANSI escape sequences for text colors
RESET = '\033[0m'
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# ANSI escape sequences for background colors
BG_BLACK = '\033[40m'
BG_RED = '\033[41m'
BG_GREEN = '\033[42m'
BG_YELLOW = '\033[43m'
BG_BLUE = '\033[44m'
BG_MAGENTA = '\033[45m'
BG_CYAN = '\033[46m'
BG_WHITE = '\033[47m'

# ANSI escape sequences for text styles
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# List of Python scripts
scripts = [
    './gpt-cli/gpt-char-exp.py',
    './gpt-cli/gpt-char.py',
    './gpt-cli/gpt-file-loader-experimental.py',
    './gpt-cli/gpt-main.py',
    './gpt-gui/gui.py'
]


def run_script(script_path):
    # Save the current working directory
    original_dir = os.getcwd()

    # Change to the directory of the script
    os.chdir(os.path.dirname(script_path))

    # Run the script
    subprocess.call(['python', os.path.basename(script_path)])

    # Change back to the original directory
    os.chdir(original_dir)


def main():
    while True:
        print("\nPython scripts:")
        for i, script in enumerate(scripts):
            if script == './gpt-gui/gui.py':
                print(f"{BLUE}{i+1}) {script} (recommended){RESET}")
            else:
                print(f"{GREEN}{i + 1}) {script}{RESET}")

        choice = input("\nChoose a script to run by entering its number (or 'q' to quit): ")
        if choice.lower() == 'q':
            break

        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(scripts):
            print("Invalid choice. Please try again.")
            continue

        script_path = scripts[int(choice) - 1]
        print(f"Running script: {script_path}\n")
        run_script(script_path)


if __name__ == "__main__":
    main()
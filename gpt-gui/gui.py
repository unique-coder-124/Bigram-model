import time
import customtkinter as ctk
import os
import tkinter
import torch
import math
import threading
import queue
import gpt_char as ai

# Set customtkinter appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Create the main application window
app = ctk.CTk()
app.geometry("500x500")
app.title("Language Model GUI")

output_queue = queue.Queue()


# Function to handle button click
def train_button_callback():
    # Get selected options from dropdowns and input values
    selected_training_file = training_file_dropdown.get()
    selected_device = device_dropdown.get()
    selected_batch_size = batch_size_slider.get()
    selected_block_size = block_size_slider.get()
    selected_max_iters = max_iters_slider.get()
    save_model_as = save_model.get()

    # Print selected options
    print("Selected Training File:", selected_training_file)
    print("Selected Device:", selected_device)
    print("Selected Batch Size:", selected_batch_size)
    print("Selected Block Size:", selected_block_size)
    print("Selected Max Iters:", selected_max_iters)
    print("Saved model as:", save_model_as)

    # Start the training on a separate thread
    train_thread = threading.Thread(target=ai.main_code, kwargs={
        'batch_size_set': selected_batch_size,
        'block_size_set': selected_block_size,
        'max_iters_set': selected_max_iters,
        'mode': 'n',
        'train_data_path': selected_training_file,
        'save_model_as': save_model_as,
        'device_set': selected_device,
        'train_progress': update_train_progress
    })
    train_thread.start()


def generate_button_callback():
    # Get selected options from dropdowns and input values
    selected_model = model_dropdown.get()
    prompt_text = prompt_input.get("0.0", "end")
    selected_device = device2_dropdown.get()
    selected_response_length = response_length_slider.get()

    # Print selected options
    print("Selected Model:", selected_model)
    print("Prompt:", prompt_text)
    print("Selected Device:", selected_device)
    print("Selected Response Length:", selected_response_length)

    # Start the generation on a separate thread
    generate_thread = threading.Thread(target=generate_thread_func, args=(output_queue, selected_model, prompt_text, selected_device, selected_response_length))
    generate_thread.start()

    # Check if the generate thread has finished and update the response output
    app.after(100, check_generate_thread, generate_thread)


def generate_thread_func(output_queue, selected_model, prompt_text, selected_device, selected_response_length):
    gpt_response = ai.main_code(
        model_load_path_set=selected_model,
        prompt=prompt_text,
        device_set=selected_device,
        max_response_set=selected_response_length,
        mode='y',
        generate_progress=update_generate_progress
    )
    output_queue.put(gpt_response)


def check_generate_thread(generate_thread):
    if generate_thread.is_alive():
        # If the generate thread is still running, check again after 100 ms
        app.after(100, check_generate_thread, generate_thread)
    else:
        # If the generate thread has finished, get the output and update the response output
        gpt_response = output_queue.get()
        update_response(gpt_response)


# Create a tabbed layout
tab_layout = ctk.CTkTabview(app)
tab_layout.pack(pady=20, padx=20, fill="both", expand=True)

# Create the "Train" tab
train_tab = tab_layout.add("Train")

# Create a frame for the train tab
train_frame = ctk.CTkScrollableFrame(train_tab)
train_frame.pack(pady=20, padx=20, fill="both", expand=True)

save_model_label = ctk.CTkLabel(master=train_frame, text="Save Model as:")
save_model_label.pack(pady=10, padx=10)
save_model = ctk.CTkEntry(master=train_frame)
save_model.pack(pady=10, padx=10)

# Create and pack GUI elements for the train tab
training_file_label = ctk.CTkLabel(train_frame, text="Training File:")
training_file_label.pack(pady=10, padx=10)
training_files = [file for file in os.listdir("../training_data") if file.endswith(".txt")]
training_file_dropdown = ctk.CTkOptionMenu(train_frame, values=training_files)
training_file_dropdown.pack(pady=10, padx=10)

device_label = ctk.CTkLabel(train_frame, text="Device:")
device_label.pack(pady=10, padx=10)
available_devices = []
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    available_devices.append("mps")
if torch.cuda.is_available():
    available_devices.append("cuda")
available_devices.append("cpu")
device_dropdown = ctk.CTkOptionMenu(train_frame, values=available_devices)
device_dropdown.pack(pady=10, padx=10)

batch_size_label = ctk.CTkLabel(train_frame, text="Batch Size:")
batch_size_label.pack(pady=10, padx=10)
batch_size_slider = ctk.CTkSlider(train_frame, from_=1, to=512)
batch_size_slider.pack(pady=10, padx=10)
batch_size_slider.set(32)
batch_size_input = ctk.CTkEntry(train_frame, placeholder_text="32")
batch_size_input.pack(pady=10, padx=10)


def update_batch_size_from_slider(value):
    rounded_value = int(round(value))
    batch_size_slider.set(rounded_value)
    batch_size_input.delete(0, 'end')
    batch_size_input.insert(0, str(rounded_value))


def update_batch_size_from_input(event):
    try:
        value = int(batch_size_input.get())
    except:
        value = batch_size_input.get()
    if not value == str:
        rounded_value = int(round(float(value)))
        batch_size_slider.set(rounded_value)
        batch_size_input.delete(0, 'end')
        batch_size_input.insert(0, str(rounded_value))


batch_size_input.bind("<Return>", update_batch_size_from_input)
batch_size_slider.bind("<ButtonRelease-1>", lambda event: update_batch_size_from_slider(batch_size_slider.get()))

block_size_label = ctk.CTkLabel(train_frame, text="Block Size:")
block_size_label.pack(pady=10, padx=10)
block_size_slider = ctk.CTkSlider(train_frame, from_=1, to=1024)
block_size_slider.pack(pady=10, padx=10)
block_size_slider.set(64)
block_size_input = ctk.CTkEntry(train_frame, placeholder_text="64")
block_size_input.pack(pady=10, padx=10)


def update_block_size_from_slider(value):
    rounded_value = int(round(value))
    block_size_slider.set(rounded_value)
    block_size_input.delete(0, 'end')
    block_size_input.insert(0, str(rounded_value))


def update_block_size_from_input(event):
    try:
        value = int(block_size_input.get())
    except:
        value = block_size_input.get()
    if not value == str:
        rounded_value = int(round(float(value)))
        block_size_slider.set(rounded_value)
        block_size_input.delete(0, 'end')
        block_size_input.insert(0, str(rounded_value))


block_size_input.bind("<Return>", update_block_size_from_input)
block_size_slider.bind("<ButtonRelease-1>", lambda event: update_block_size_from_slider(block_size_slider.get()))

max_iters_label = ctk.CTkLabel(train_frame, text="Max Iters:")
max_iters_label.pack(pady=10, padx=10)
max_iters_slider = ctk.CTkSlider(train_frame, from_=100, to=100000)
max_iters_slider.pack(pady=10, padx=10)
max_iters_slider.set(5000)
max_iters_input = ctk.CTkEntry(train_frame, placeholder_text="5000")
max_iters_input.pack(pady=10, padx=10)


# Link the max_iters slider and input
def update_max_iters_from_slider(value):
    rounded_value = int(round(value))
    max_iters_slider.set(rounded_value)
    max_iters_input.delete(0, 'end')
    max_iters_input.insert(0, str(rounded_value))


def update_max_iters_from_input(event):
    try:
        value = int(max_iters_input.get())
    except:
        value = max_iters_input.get()
    if not value == str:
        rounded_value = int(round(float(value)))
        max_iters_slider.set(rounded_value)
        max_iters_input.delete(0, 'end')
        max_iters_input.insert(0, str(rounded_value))


max_iters_input.bind("<Return>", update_max_iters_from_input)
max_iters_slider.bind("<ButtonRelease-1>", lambda event: update_max_iters_from_slider(max_iters_slider.get()))

train_progress = ctk.CTkProgressBar(master=train_frame, height=10, mode='determinate')
train_progress.pack(pady=10, padx=10)
train_progress.set(0)


def update_train_progress(value):
    train_progress.set(value)


train_button = ctk.CTkButton(train_frame, text="Train", command=train_button_callback)
train_button.pack(pady=10, padx=10)

# Create the "Generate" tab
generate_tab = tab_layout.add("Generate")

# Create a frame for the generate tab
generate_frame = ctk.CTkScrollableFrame(generate_tab)
generate_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Create and pack GUI elements for the generate tab
model_default = ctk.StringVar(value='model_char.pt')
model_label = ctk.CTkLabel(master=generate_frame, text="Model:")
model_label.pack(pady=10, padx=10)
model_files = [file for file in os.listdir("../models") if file.endswith(".pt")]
model_dropdown = ctk.CTkOptionMenu(master=generate_frame, values=model_files, variable=model_default)
model_dropdown.pack(pady=10, padx=10)

prompt_label = ctk.CTkLabel(master=generate_frame, text="Prompt:")
prompt_label.pack(pady=10, padx=10)
prompt_input = ctk.CTkTextbox(master=generate_frame)
prompt_input.pack(pady=10, padx=10)

response_label = ctk.CTkLabel(master=generate_frame, text="Response:")
response_label.pack(pady=10, padx=10)

response_frame = ctk.CTkFrame(master=generate_frame)
response_frame.pack(pady=10, padx=10)

response_output = ctk.CTkLabel(master=response_frame, width=40, height=10, anchor="nw")
response_output.pack(pady=10, padx=10)
# response_output.configure(background="#2b2b2b")
response_output.configure(text="")
# response_output.configure(foreground="#FFFFFF")


def update_response(response):
    response_output.configure(text="")
    response_output.configure(text=response)


device2_label = ctk.CTkLabel(master=generate_frame, text="Device:")
device2_label.pack(pady=10, padx=10)
device2_dropdown = ctk.CTkOptionMenu(master=generate_frame, values=available_devices)
device2_dropdown.pack(pady=10, padx=10)

response_length_label = ctk.CTkLabel(master=generate_frame, text="Response Length:")
response_length_label.pack(pady=10, padx=10)
response_length_slider = ctk.CTkSlider(master=generate_frame, from_=128, to=2048)
response_length_slider.pack(pady=10, padx=10)
response_length_slider.set(512)

response_length_input = ctk.CTkEntry(generate_frame, placeholder_text="512")
response_length_input.pack(pady=10, padx=10)


def update_response_length_from_slider(value):
    rounded_value = int(round(value))
    response_length_slider.set(rounded_value)
    response_length_input.delete(0, 'end')
    response_length_input.insert(0, str(rounded_value))


def update_response_length_from_input(event):
    try:
        value = int(response_length_input.get())
    except:
        value = response_length_input.get()
    if not value == str:
        rounded_value = int(round(float(value)))
        response_length_slider.set(rounded_value)
        response_length_input.delete(0, 'end')
        response_length_input.insert(0, str(rounded_value))


response_length_input.bind("<Return>", update_response_length_from_input)
response_length_slider.bind("<ButtonRelease-1>",
                            lambda event: update_response_length_from_slider(response_length_slider.get()))

generate_progress = ctk.CTkProgressBar(master=generate_frame, height=10, mode='determinate')
generate_progress.pack(pady=10, padx=10)
generate_progress.set(0)


# Function to update the progress bar
def update_generate_progress(value):
    generate_progress.set(value)


generate_button = ctk.CTkButton(master=generate_frame, text="Generate", command=generate_button_callback)
generate_button.pack(pady=10, padx=10)

# Start the application main loop
if __name__ == '__main__':
    app.mainloop()

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import threading
 
 
# Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
 
 
# Define image dimensions and number of classes
IMG_HEIGHT, IMG_WIDTH = 30, 30
NUM_CLASSES = 43
 
 
# Define model path
MODEL_PATH = 'traffic_sign_model.keras'
 
 
# Class names for GTSRB (German Traffic Sign Recognition Benchmark)
# Ensure that these names match your dataset's classes. Update if necessary.
class_names = [
  'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60',
  'Speed Limit 70', 'Speed Limit 80', 'End of Speed Limit 80', 'Speed Limit (End)',
  'Speed Limit (End)', 'Right-of-Way at Intersection', 'Priority Road',
  'Give Way', 'Stop', 'No Traffic Both Ways', 'No Trucks',
  'No Entry', 'General Caution', 'Dangerous Curve Left', 'Dangerous Curve Right',
  'Double Curve', 'Bumpy Road', 'Slippery Road', 'Road Narrows on the Right',
  'Road Work', 'Traffic Signals', 'Pedestrians', 'Children Crossing',
  'Bicycles Crossing', 'Beware of Ice/Snow', 'Wild Animals Crossing',
  'End of All Speed and Passing Limits', 'Turn Right Ahead',
  'Turn Left Ahead', 'Ahead Only', 'Go Straight or Right', 'Go Straight or Left',
  'Keep Right', 'Keep Left', 'Roundabout Mandatory', 'End of No Passing',
  'End of No Passing by Vehicles over 3.5 Tons', 'No Stopping', 'No Parking',
  'No Standing'
]
 
 
def load_data(data_dir):
  """
  Load images and labels from the specified directory.
 
 
  Args:
      data_dir (str): Path to the training data directory.
 
 
  Returns:
      Tuple of NumPy arrays: (images, labels)
  """
  images, labels = [], []
  for class_dir in os.listdir(data_dir):
      class_path = os.path.join(data_dir, class_dir)
      if not os.path.isdir(class_path):
          continue
      try:
          label = int(class_dir)
      except ValueError:
          print(f"Skipping non-integer directory: {class_dir}")
          continue
      for img_name in os.listdir(class_path):
          # Filter out non-image files
          if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
              continue  # Skip non-image files like CSVs
          img_path = os.path.join(class_path, img_name)
          img = cv2.imread(img_path)
          if img is None:
              print(f"Warning: Unable to read image {img_path}. Skipping.")
              continue
          img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          images.append(img)
          labels.append(label)
  return np.array(images), np.array(labels)
 
 
def preprocess_data(X, y):
  """
  Normalize image data and convert labels to categorical format.
 
 
  Args:
      X (np.array): Array of images.
      y (np.array): Array of labels.
 
 
  Returns:
      Tuple of NumPy arrays: (X_processed, y_processed)
  """
  X = X.astype('float32') / 255.0
  y = to_categorical(y, NUM_CLASSES)
  return X, y
 
 
def build_model(input_shape, num_classes):
  """
  Build and return a CNN model for traffic sign classification.
 
 
  Args:
      input_shape (tuple): Shape of the input images (height, width, channels).
      num_classes (int): Number of classes.
 
 
  Returns:
      keras.Model: Compiled CNN model.
  """
  inputs = layers.Input(shape=input_shape)
  x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D((2, 2))(x)
 
 
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D((2, 2))(x)
 
 
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.Dense(num_classes, activation='softmax')(x)
 
 
  model = models.Model(inputs=inputs, outputs=outputs)
  return model
 
 
def save_model(model, path=MODEL_PATH):
  """
  Save the trained model to the specified path.
 
 
  Args:
      model (keras.Model): Trained Keras model.
      path (str): File path to save the model.
  """
  model.save(path)
 
 
class TrainingCallback(tf.keras.callbacks.Callback):
  """
  Custom callback to update the GUI with training progress and logs.
  """
 
 
  def __init__(self, app):
      super().__init__()
      self.app = app
 
 
  def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}
      progress = ((epoch + 1) / self.params['epochs']) * 100
      message = (f"Epoch {epoch + 1}/{self.params['epochs']} - "
                 f"Loss: {logs.get('loss', 0):.4f} - "
                 f"Accuracy: {logs.get('accuracy', 0):.4f} - "
                 f"Val_Loss: {logs.get('val_loss', 0):.4f} - "
                 f"Val_Accuracy: {logs.get('val_accuracy', 0):.4f}\n")
      self.app.update_progress(progress)
      self.app.log_message(message)
 
 
  def on_train_end(self, logs=None):
      self.app.log_message("Training completed.\n")
      self.app.enable_train_buttons()
      messagebox.showinfo("Training Complete", f"Model trained and saved as {MODEL_PATH}.")
 
 
class TrafficSignApp:
  def __init__(self, master):
      self.master = master
      master.title("Traffic Sign Detection and Training - The Pycodes")
      master.geometry("800x800")  # Increased height to accommodate scrollbars
 
 
      self.model = None
 
 
      # Initialize UI components
      self.create_widgets()
 
 
  def create_widgets(self):
      # Frame for Training
      training_frame = tk.LabelFrame(self.master, text="Model Training", padx=10, pady=10)
      training_frame.pack(fill="both", expand="yes", padx=10, pady=10)
 
 
      # Training Directory Selection
      self.train_dir_label = tk.Label(training_frame, text="Training Directory: Not Selected", wraplength=600,
                                      justify="left")
      self.train_dir_label.pack(anchor='w')
 
 
      self.select_train_dir_button = tk.Button(training_frame, text="Select Training Directory",
                                               command=self.select_train_directory)
      self.select_train_dir_button.pack(pady=5)
 
 
      # Start Training Button
      self.train_button = tk.Button(training_frame, text="Start Training", command=self.start_training,
                                    state='disabled', bg='green', fg='white')
      self.train_button.pack(pady=5)
 
 
      # Progress Bar
      self.progress_var = tk.DoubleVar()
      self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var, maximum=100)
      self.progress_bar.pack(fill='x', pady=5)
 
 
      # Training Logs
      self.log_text = scrolledtext.ScrolledText(training_frame, height=10, state='disabled')
      self.log_text.pack(fill='both', expand=True, pady=5)
 
 
      # Separator
      separator = tk.Frame(self.master, height=2, bd=1, relief='sunken')
      separator.pack(fill='x', padx=5, pady=10)
 
 
      # Frame for Prediction
      prediction_frame = tk.LabelFrame(self.master, text="Traffic Sign Prediction", padx=10, pady=10)
      prediction_frame.pack(fill="both", expand="yes", padx=10, pady=10)
 
 
      # Create a Canvas inside prediction_frame
      canvas = tk.Canvas(prediction_frame, borderwidth=0, background="#f0f0f0")
      canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
 
 
      # Add vertical and horizontal scrollbars to the Canvas
      v_scrollbar = tk.Scrollbar(prediction_frame, orient=tk.VERTICAL, command=canvas.yview)
      v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
 
      h_scrollbar = tk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=canvas.xview)
      h_scrollbar.pack(fill=tk.X)
 
 
      canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
 
 
      # Create an inner frame to hold prediction widgets
      self.inner_prediction_frame = tk.Frame(canvas, background="#f0f0f0")
      canvas.create_window((0, 0), window=self.inner_prediction_frame, anchor='nw')
 
 
      # Bind the inner frame to configure the scroll region
      self.inner_prediction_frame.bind("<Configure>", lambda event, canvas=canvas: self.on_frame_configure(canvas))
 
 
      # Upload Image Button
      self.upload_button = tk.Button(self.inner_prediction_frame, text="Upload Image for Sign Recognition", command=self.upload_image,
                                     bg='blue', fg='white', font=('Helvetica', 12, 'bold'))
      self.upload_button.pack(pady=10)
 
 
      # Create a frame to hold the canvas and scrollbars for the image
      self.image_frame = tk.Frame(self.inner_prediction_frame)
      self.image_frame.pack(pady=5, fill=tk.BOTH, expand=True)
 
 
      # Create Canvas for image display
      self.image_canvas = tk.Canvas(self.image_frame, width=400, height=300, bg='gray')
      self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
 
 
      # Add scrollbars to the image_canvas
      img_v_scrollbar = tk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
      img_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
      img_h_scrollbar = tk.Scrollbar(self.inner_prediction_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
      img_h_scrollbar.pack(fill=tk.X)
 
 
      self.image_canvas.configure(yscrollcommand=img_v_scrollbar.set, xscrollcommand=img_h_scrollbar.set)
 
 
      # Prediction Label
      self.prediction_label = tk.Label(self.inner_prediction_frame, text="Traffic Sign Recognition: None", font=('Helvetica', 14))
      self.prediction_label.pack(pady=5)
 
 
      # Confidence Label
      self.confidence_label = tk.Label(self.inner_prediction_frame, text="Confidence: 0.00%", font=('Helvetica', 14))
      self.confidence_label.pack(pady=5)
 
 
  def on_frame_configure(self, canvas):
      """
      Reset the scroll region to encompass the inner frame
      """
      canvas.configure(scrollregion=canvas.bbox("all"))
 
 
  def select_train_directory(self):
      directory = filedialog.askdirectory()
      if directory:
          self.train_dir = directory
          self.train_dir_label.config(text=f"Training Directory: {directory}")
          self.train_button.config(state='normal')
      else:
          self.train_dir_label.config(text="Training Directory: Not Selected")
          self.train_button.config(state='disabled')
 
 
  def start_training(self):
      # Disable buttons to prevent multiple training sessions
      self.train_button.config(state='disabled')
      self.select_train_dir_button.config(state='disabled')
      self.log_text.config(state='normal')
      self.log_text.insert(tk.END, "Starting training...\n")
      self.log_text.config(state='disabled')
      self.progress_var.set(0)
 
 
      # Start training in a separate thread
      training_thread = threading.Thread(target=self.train_model)
      training_thread.start()
 
 
  def train_model(self):
      try:
          # Load and preprocess data
          self.log_message("Loading training data...\n")
          X, y = load_data(self.train_dir)
          if X.size == 0:
              self.log_message("No data found. Please check the training directory.\n")
              self.enable_train_buttons()
              return
          self.log_message(f"Loaded {X.shape[0]} images.\n")
 
 
          self.log_message("Preprocessing data...\n")
          X, y = preprocess_data(X, y)
 
 
          # Split into training and validation sets
          self.log_message("Splitting data into training and validation sets...\n")
          X_train, X_val, y_train, y_val = train_test_split(
              X, y, test_size=0.2, random_state=42
          )
          self.log_message(f"Training samples: {X_train.shape[0]}\n")
          self.log_message(f"Validation samples: {X_val.shape[0]}\n")
 
 
          # Data Augmentation
          self.log_message("Applying data augmentation...\n")
          datagen = ImageDataGenerator(
              rotation_range=10,
              zoom_range=0.1,
              width_shift_range=0.1,
              height_shift_range=0.1,
              horizontal_flip=False
          )
          datagen.fit(X_train)
 
 
          # Build the model
          self.log_message("Building the CNN model...\n")
          self.model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES)
          self.log_message("Model architecture:\n")
          model_summary = []
          self.model.summary(print_fn=lambda x: model_summary.append(x))
          self.log_message("\n".join(model_summary) + "\n")
 
 
          # Compile the model
          self.log_message("Compiling the model...\n")
          self.model.compile(
              optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
          )
 
 
          # Define callbacks
          training_callback = TrainingCallback(self)
          early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
 
 
          # Train the model with EarlyStopping
          self.log_message("Starting training...\n")
          self.model.fit(
              datagen.flow(X_train, y_train, batch_size=32),
              epochs=20,
              validation_data=(X_val, y_val),
              callbacks=[training_callback, early_stopping],
              verbose=0  # Suppress Keras' own output
          )
 
 
          # Save the model
          save_model(self.model)
          self.log_message(f"Model trained and saved as {MODEL_PATH}.\n")
 
 
      except Exception as e:
          self.log_message(f"Error during training: {e}\n")
          messagebox.showerror("Training Error", str(e))
          self.enable_train_buttons()
 
 
  def log_message(self, message):
      # Schedule GUI update in the main thread
      self.master.after(0, self._log_message, message)
 
 
  def _log_message(self, message):
      self.log_text.config(state='normal')
      self.log_text.insert(tk.END, message)
      self.log_text.see(tk.END)
      self.log_text.config(state='disabled')
 
 
  def update_progress(self, progress):
      # Schedule GUI update in the main thread
      self.master.after(0, self._update_progress, progress)
 
 
  def _update_progress(self, progress):
      self.progress_var.set(progress)
      self.master.update_idletasks()
 
 
  def enable_train_buttons(self):
      # Schedule GUI update in the main thread
      self.master.after(0, self._enable_train_buttons)
 
 
  def _enable_train_buttons(self):
      self.train_button.config(state='normal')
      self.select_train_dir_button.config(state='normal')
 
 
  def upload_image(self):
      file_path = filedialog.askopenfilename(
          filetypes=[
              ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.ppm"),
              ("All Files", "*.*")
          ]
      )
      if not file_path:
          return  # User cancelled
 
 
      # Display the image with scrollbars
      try:
          img = Image.open(file_path)
          self.original_img = img.copy()  # Keep a copy for potential future use
 
 
          # Do not resize the image for display to enable scrolling
          img_tk = ImageTk.PhotoImage(img)
          self.image_canvas.delete("all")
          self.image_canvas.create_image(0, 0, anchor='nw', image=img_tk)
          self.image_canvas.image = img_tk  # Keep a reference
 
 
          # Update the scrollregion to the size of the image
          self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL))
      except Exception as e:
          messagebox.showerror("Error", f"Unable to open image.\n{e}")
          return
 
 
      # Predict the class
      if not self.model:
          if os.path.exists(MODEL_PATH):
              try:
                  self.log_message("Loading trained model for prediction...\n")
                  self.model = tf.keras.models.load_model(MODEL_PATH)
                  self.log_message("Model loaded successfully.\n")
              except Exception as e:
                  messagebox.showerror("Error", f"Failed to load model.\n{e}")
                  return
          else:
              messagebox.showerror("Model Not Found",
                                   f"Model file '{MODEL_PATH}' not found. Please train the model first.")
              return
 
 
      class_id, confidence = self.predict_image(file_path)
      if class_id is not None:
          if class_id < len(class_names):
              predicted_class = class_names[class_id]
          else:
              predicted_class = f"Unknown Class ({class_id})"
          confidence_percent = confidence * 100
          self.prediction_label.config(text=f"Prediction: {predicted_class}")
          self.confidence_label.config(text=f"Confidence: {confidence_percent:.2f}%")
      else:
          self.prediction_label.config(text="Prediction: Error")
          self.confidence_label.config(text="Confidence: N/A")
 
 
  def preprocess_image(self, image_path):
      """
      Preprocess the image for prediction.
 
 
      Args:
          image_path (str): Path to the image file.
 
 
      Returns:
          np.array: Preprocessed image ready for prediction.
      """
      img = cv2.imread(image_path)
      if img is None:
          raise ValueError(f"Unable to read image at {image_path}")
      img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = img.astype('float32') / 255.0
      img = np.expand_dims(img, axis=0)  # Add batch dimension
      return img
 
 
  def predict_image(self, image_path):
      """
      Preprocess the image, run the model prediction, and return the predicted class and confidence score.
 
 
      Args:
          image_path (str): Path to the image file.
 
 
      Returns:
          Tuple: (class_id, confidence_score)
      """
      try:
          img = self.preprocess_image(image_path)
      except Exception as e:
          messagebox.showerror("Error", str(e))
          return None, None
 
 
      predictions = self.model.predict(img)
      class_id = np.argmax(predictions, axis=1)[0]
      confidence = np.max(predictions)
      return class_id, confidence
 
 
if __name__ == "__main__":
  root = tk.Tk()
  app = TrafficSignApp(root)
  root.mainloop()
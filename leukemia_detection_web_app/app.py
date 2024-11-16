import os
import torch
from flask import Flask, request, render_template
from src.preprocess import preprocess_image
from src.model import build_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define allowed file extensions and upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','tif'}
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = build_model(num_classes=3)
model.load_state_dict(torch.load('leukemia_model.pth'))
model.eval()

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page (upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        img_tensor = preprocess_image(file_path)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(img_tensor.device)

        # Make the prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)

        # Map the predicted class to a label (L1, L2, L3)
        class_labels = {0: 'Leukemia Type 1', 1: 'Leukemia Type 2', 2: 'Leukemia Type 3'}
        predicted_label = class_labels[predicted_class.item()]
        
        return render_template('index.html', prediction=predicted_label, filename=filename)
    return 'Invalid file format'

if __name__ == "__main__":
    app.run(debug=True)

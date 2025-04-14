import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from data_balancer import DatasetBalancer

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        target_column = request.form.get('target_column', 'target')
        
        # If user does not select file, browser submits empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            try:
                # Process the file
                balancer = DatasetBalancer(target_column=target_column)
                shape, class_dist = balancer.load_data(upload_path)
                balancer.preprocess()
                balanced_df = balancer.balance()
                
                # Save processed file
                processed_filename = f"balanced_{filename}"
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                balanced_df.to_csv(processed_path, index=False)
                
                return render_template('index.html', 
                                    original_shape=shape,
                                    original_dist=class_dist,
                                    processed_file=processed_filename,
                                    target_column=target_column)
            
            except Exception as e:
                flash(str(e))
                return redirect(request.url)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
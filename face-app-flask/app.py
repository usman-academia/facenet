from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename

from bin.recognition import (
    load_model,
    compute_embedding,
    find_most_similar,
    save_embedding
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model once globally
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # IDENTIFY logic (unchanged)
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        embedding = compute_embedding(filepath, model)
        name = find_most_similar(embedding)

        return render_template(
            'index.html',
            name=name,
            user_image=filename
        )

    # GET -> show identify + save UI
    return render_template(
        'index.html',
        name=None,
        user_image=None
    )

@app.route('/save', methods=['POST'])
def save_user():
    # 1. Grab the username and file
    username = request.form.get('username', '').strip()
    file = request.files.get('file')
    if not username or not file or file.filename == '':
        # Could add flash() messages here
        return redirect(url_for('index'))

    # 2. Save the file locally
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 3. Compute embedding and persist it
    embedding = compute_embedding(filepath, model)
    # save_embedding should accept (username, embedding)
    save_embedding(username, embedding)

    # 4. Redirect back (or to a thank-you page)
    return render_template(
        'index.html',
        saved_username=username,
        saved_image=filename
    )

if __name__ == '__main__':
    app.run(debug=True)

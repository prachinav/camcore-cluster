from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from utils import plot_cluster_diagnostics, run_clustering_pipeline
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csv_file']
        if not file:
            return "No file uploaded", 400

        df = pd.read_csv(file)
        df.to_csv(os.path.join(UPLOAD_FOLDER, 'uploaded.csv'), index=False)

        silhouette_path, elbow_path, db_path = plot_cluster_diagnostics(df, upload_dir=UPLOAD_FOLDER)

        return render_template(
            'index.html',
            silhouette_img=silhouette_path,
            elbow_img=elbow_path,
            db_img=db_path,
            show_k_input=True
        )

    return render_template('index.html', show_k_input=False)

@app.route('/cluster', methods=['POST'])
def cluster():
    k = int(request.form['k'])
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'uploaded.csv'))

    results = run_clustering_pipeline(
        csv_path=os.path.join(UPLOAD_FOLDER, 'uploaded.csv'),
        k=k,
        output_dir=UPLOAD_FOLDER
    )

    return render_template(
        'index.html',
        pca_img=results["pca_img"],
        map_future=results["map_future"],
        map_past=results["map_past"],
        map_transition=results["map_transition"],
        show_k_input=False
    )


if __name__ == '__main__':
    app.run(debug=True)

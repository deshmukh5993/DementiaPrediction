import bcrypt
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, UserMixin
from flask_bcrypt import Bcrypt
import pickle
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sqlite3
from collections import Counter
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import send_file
from flask import jsonify

with open('model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

app = Flask(__name__, static_url_path='/static')
bcrypt = Bcrypt(app)
app.debug = True


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'thisissecret'
app.app_context().push()

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    fname = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f"User('{self.id}','{self.fname}','{self.username}','{self.password}')"


class User_Details(db.Model):
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    Name = db.Column(db.String(20), nullable=False)
    Gender = db.Column(db.Integer, nullable=False)
    Hand = db.Column(db.Integer, nullable=False)
    Age = db.Column(db.Integer, nullable=False)
    Educ = db.Column(db.Integer, nullable=True)
    SES = db.Column(db.Float, nullable=False)
    MMSE = db.Column(db.Float, nullable=False)
    CDR = db.Column(db.Float, nullable=False)
    eTIV = db.Column(db.Integer, nullable=False)
    nWBV = db.Column(db.Float, nullable=False)
    ASF = db.Column(db.Float, nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    Result = db.Column(db.VARCHAR(80), nullable=False)

    def __repr__(self):
        return f"User_Details('{self.id}','{self.Name}','{self.date_added}','{self.Result}')"


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user:
            if bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect('/dboard')

    return render_template('login.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        Name = request.form.get('name')
        Gender = request.form.get('gender')
        Hand = request.form.get('hand')
        Age = request.form.get('age')
        Educ = request.form.get('educ')
        SES = request.form.get('ses')
        MMSE = request.form.get('mmse')
        CDR = request.form.get('cdr')
        eTIV = request.form.get('eTIV')
        nWBV = request.form.get('nWBV')
        ASF = request.form.get('ASF')

        input_data = [Gender, Hand, Age, Educ, SES, MMSE, CDR, eTIV, nWBV, ASF]
        input_data_np = np.asarray(input_data)
        input_data_re = input_data_np.reshape(1, -1)

        pred = loaded_data.predict(input_data_re)
        print(pred)
        pred1 = pred

        if pred1[0] == 1:
            result1 = "Demented"
            user_details = User_Details(Name=Name, Gender=Gender, Hand=Hand, Age=Age, Educ=Educ, SES=SES, MMSE=MMSE,
                                        CDR=CDR, eTIV=eTIV, nWBV=nWBV, ASF=ASF, Result=result1)
            db.session.add(user_details)
            db.session.commit()
            print("d "+str(pred1[0]))
            # return result1 + str(pred1[0])
            return render_template('Result.html', predict_content=result1)

        else:
            result2 = "Non-demented"
            user_details = User_Details(Name=Name, Gender=Gender, Hand=Hand, Age=Age, Educ=Educ, SES=SES, MMSE=MMSE,
                                        CDR=CDR, eTIV=eTIV, nWBV=nWBV, ASF=ASF, Result=result2)
            db.session.add(user_details)
            db.session.commit()
            print("nd  "+str(pred1[0]))
            # return result2 + str(pred1[0])
            return render_template('Result.html', predict_content=result2)

    return render_template('form.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fname = request.form.get('fname')
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_pass = bcrypt.generate_password_hash(password)
        user = User(fname=fname, username=username, password=hashed_pass)
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')


@app.route('/dboard', methods=['GET', 'POST'])
def dboard():
    return render_template('dashboard.html')


@app.route('/patient_details', methods=['GET', 'POST'])
def patient_details():

    data = User_Details.query.all()
    return render_template('patient_details.html', data=data)


@app.route('/plot_graph')
def plot_graph():
    data = User_Details.query.all()

    # Extract data from columns
    group_values = [item.Result for item in data]

    # Count occurrences of each group
    group_counts = Counter(group_values)

    # Calculate percentages
    total_groups = len(group_values)
    percentages = [(Result, count / total_groups * 100)
                   for Result, count in group_counts.items()]

    # Separate the percentages into x and y values
    x_values, y_values = zip(*percentages)

    bar_colors = ['red', 'green']

    # Plot the bar graph using Matplotlib
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].bar(x_values, y_values, color=bar_colors)
    ax[0].set_xlabel('Result')
    ax[0].set_ylabel('Percentage')
    ax[0].set_title('Result Percentage Bar Graph')

    color_map = {'Demented': 'red', 'Non-demented': 'green'}
    # Pie Chart
    ax[1].pie(y_values, labels=x_values, autopct='%1.1f%%',
              startangle=90, colors=[color_map[result] for result in x_values])
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[1].axis('equal')
    ax[1].set_title('Group Percentage Pie Chart')

    # Convert the Matplotlib figure to a PNG image
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')


def generate_plot():
    plot_image = plot_graph()
    return render_template('plot_graph.html', plot_image=plot_image)


@app.route('/logout')
def logout():
    return redirect('login.html')


@app.route('/search_by_id', methods=['GET'])
def search_by_id():
    search_id = request.args.get('searchId')

    # Assuming id is an integer in User_Details
    try:
        search_id = int(search_id)
    except ValueError:
        return jsonify({'error': 'Invalid ID format'})

    # Query the database to get the User_Details with the specified ID
    filtered_data = User_Details.query.filter_by(id=search_id).all()

    return render_template('patient_details.html', data=filtered_data)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask
from app import views

app = Flask(__name__) # webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/person_list/',
                 endpoint='person_list',
                 view_func=views.person_list)
app.add_url_rule(rule='/fatigue_analysis/',
                 endpoint='fatigue_analysis',
                 view_func=views.fatigue_analysis,
                 methods=['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)
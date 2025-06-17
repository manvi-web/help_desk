from flask import Flask, session
from flask_session import Session

app = Flask(__name__)
app.secret_key = "secret"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
def home():
    session["counter"] = session.get("counter", 0) + 1
    return f"Counter = {session['counter']}"

if __name__ == "__main__":
    app.run(debug=True, port=5050)

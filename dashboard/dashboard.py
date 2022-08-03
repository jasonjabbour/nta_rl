from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def dashboard_home():
    return render_template('index.html',
                           network_traffic=[1]*15, 
                           rl_confidence=[2]*15, 
                           true_network_state=1, 
                           rl_alert=1)

if __name__ == '__main__':
    app.run(debug=True)
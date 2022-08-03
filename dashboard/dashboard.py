import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from flask import Flask, render_template
from digital_twin import DigitalTwin

app = Flask(__name__)


@app.route('/')
def dashboard_home():
    global run
    global obs
    global model
    global step
    global previous_rl_alert
    
    # Let alert sit on screen for longer
    if previous_rl_alert:
        time.sleep(5)
    
    # Get prediction for previous observation
    action, obs, step = run.callable_test(model, obs, step )
    # Format action and observations
    action = [round(num, 2) for num in action]
    obs = [round(num, 2) for num in obs] 
    
    # Determine true state of network
    true_network_state = run.get_under_attack()
    if true_network_state:
        true_network_state = 1
    else:
        true_network_state = 0

    # Determine if alert should be sent based on policy confidence
    if action[-1] > .8:
        rl_alert = 1
    else:
        rl_alert = 0
        
    step+=1
    previous_rl_alert = rl_alert
    
    # Refresh variables within HTML script
    return render_template('index.html',
                           network_traffic=obs[:15], 
                           rl_confidence=action[:15], 
                           true_network_state=true_network_state, 
                           rl_alert=rl_alert)

if __name__ == '__main__':
    # Initialize Digital Twin
    run = DigitalTwin(mode='None', 
                      best_model=True)
    model = run.load_trained_policy()
    obs = run.reset_test_env()
    step = 0 
    previous_rl_alert = 0
    
    app.run(debug=True)

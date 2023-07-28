from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np
import time
import math
import redis
import json
from threading import Thread, Event

# Load the model
model = tf.keras.models.load_model('my_model')

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # enable CORS for SocketIO

# Initialize the redis client
r = redis.Redis()

# Initialize a publisher
pubsub = r.pubsub()

# Global variable to control the state of data generation
generate_abnormal = False


def consumer():
    while True:
        # Use Redis' lpop function to retrieve and remove the left most item from the list
        message = r.lpop('ecg_data')

        # If the message is None, then the list is empty. Wait for a second before trying again
        if message is None:
            time.sleep(1)
            continue

        message = json.loads(message)
        incoming = np.array(message['data']).reshape((1, 140))
        predictions = model.predict(incoming)
        flattened_predictions = predictions.tolist()[0]

        # Send the prediction to the dashboard
        socketio.emit(f'new_prediction', {
            'prediction': flattened_predictions, 'label': message["label"]
        })


# Create threads for consumer
consumer_thread = Thread(target=consumer)
consumer_thread.start()


@app.route('/')
def index():
    return render_template('index.html')  # an HTML page with your graph code


# Create an Event object
generate_abnormal_event = Event()


@app.route('/trigger', methods=['GET'])
def trigger():
    # Toggle the event state
    if r.get('generate_abnormal') == b"1":
        r.set('generate_abnormal', "0")
    else:
        r.set('generate_abnormal', "1")

    # Clear the 'ecg_data' queue
    r.delete('ecg_data')

    return jsonify({"status": "Abnormal data generation toggled."})


def generate_and_emit_data():
    i = 0
    while True:
        # Generate a simple sine wave as the ECG data
        ecg_data = [math.sin(x / 10.0) for x in range(i, i + 140)]

        # Modify the data if it is meant to be abnormal
        if r.get('generate_abnormal') == b"1":
            # Introduce a sudden spike more often and of greater magnitude
            if i % 10 == 0:
                ecg_data[i % 140] *= 10

            # Introduce a sudden drop more often and of greater magnitude
            if i % 30 == 0:
                ecg_data[i % 140] = -2

        # Add the data and label to the message
        message = {
            'data': ecg_data,
            'label': 'abnormal' if r.get('generate_abnormal') == b"1" else 'normal'
        }

        r.rpush('ecg_data', json.dumps(message))

        i += 1
        time.sleep(1)


# Start the data generation thread
Thread(target=generate_and_emit_data).start()

if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)

from flask import Flask, render_template, request, jsonify
from chat import get_response  # Ensure chat.py exists and has get_response function

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")  
def predict():
    text = request.get_json().get("message")
    
    if not text:  # Check if text is empty or None
        return jsonify({"answer": "Please enter a valid message."})

    response = get_response(text)
    message = {"answer": response}
    
    return jsonify(message)

if __name__ == "__main__":  
    app.run(debug=True)

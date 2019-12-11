from flask import Flask, request, redirect, jsonify
import tensorflow as tf
import numpy as np

export_path = "./exported-model"

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def detect_text():
    if request.method == "GET":
        return app.send_static_file('./index.html')

    if request.method == "POST":
        sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(sess, ["serve"], export_path)

        files = request.files.getlist("image")
        send_res = {"response":[]}
        images = []
        filenames = []
        for img in files:
            image = img.read()
            images.append(image)
            filenames.append(img.filename)
        
        out = sess.run(['prediction:0', 'probability:0'], feed_dict={'input_image_as_bytes:0': images}) 
        # Returns a list of two lists for pred and prob
        
                
        # Cannot zip non-lists so making list of single value when out[1] is not a list
        if not type(out[1]) == np.ndarray:
        	out[0] = [out[0]]
        	out[1] = [out[1]]

        for img_name, pred, prob in zip(filenames, out[0], out[1]):
            temp = {"filename":img_name, "prediction":pred.decode("utf-8"), "probability":prob}
            send_res["response"].append(temp)

        sess.close()

        return jsonify(send_res)

if __name__ == "__main__":
   app.run(debug=True)

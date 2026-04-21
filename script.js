const BASE_URL = "http://127.0.0.1:8000";

// Predict
async function predict() {
    try {
        const data = {
            sepal_length: parseFloat(sl.value),
            sepal_width: parseFloat(sw.value),
            petal_length: parseFloat(pl.value),
            petal_width: parseFloat(pw.value)
        };

        const res = await fetch(BASE_URL + "/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await res.json();

        document.getElementById("result").innerText =
            "Prediction: " + result.prediction;

        document.getElementById("confidence").innerText =
            "Confidence: " + result.confidence;

    } catch (err) {
        alert("Error: " + err);
    }
}


// Get model info
async function getModelInfo() {
    const res = await fetch(BASE_URL + "/model-info");
    const data = await res.json();

    document.getElementById("info").innerText =
        JSON.stringify(data, null, 2);
}
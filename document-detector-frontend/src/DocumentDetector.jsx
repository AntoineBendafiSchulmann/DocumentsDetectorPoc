import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const DocumentDetector = () => {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const fileInputRef = useRef();

  useEffect(() => {
    const loadModel = async () => {
      const modelUrl = `${process.env.PUBLIC_URL}/../js_model/model.json`;
      const loadedModel = await tf.loadLayersModel(modelUrl);
      setModel(loadedModel);
    };
    loadModel();
  }, []);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file && model) {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.onload = async () => {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .div(tf.scalar(255.0))
          .expandDims();
        const prediction = model.predict(tensor);
        const result = prediction.dataSync()[0];
        setPrediction(result);
      };
    }
  };

  return (
    <div>
      <h1>Document Detector</h1>
      <input type="file" ref={fileInputRef} onChange={handleImageUpload} />
      {prediction !== null && <p>Prediction: {prediction}</p>}
    </div>
  );
};

export default DocumentDetector;

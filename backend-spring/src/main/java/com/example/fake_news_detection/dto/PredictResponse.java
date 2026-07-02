package com.example.fake_news_detection.dto;

import java.util.Map;

public class PredictResponse {

    private String prediction;
    private Map<String, Double> probs;

    public PredictResponse() {}

    public PredictResponse(String prediction, Map<String, Double> probs) {
        this.prediction = prediction;
        this.probs = probs;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public Map<String, Double> getProbs() {
        return probs;
    }

    public void setProbs(Map<String, Double> probs) {
        this.probs = probs;
    }
}

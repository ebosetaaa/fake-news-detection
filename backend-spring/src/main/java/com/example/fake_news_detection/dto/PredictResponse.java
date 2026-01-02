package com.example.fake_news_detection.dto;

import java.util.List;

public class PredictResponse {

    private String prediction;
    private List<Double> probs;

    public PredictResponse() {}

    public PredictResponse(String prediction, List<Double> probs) {
        this.prediction = prediction;
        this.probs = probs;
    }

    public String getPrediction() { return prediction; }
    public void setPrediction(String prediction) { this.prediction = prediction; }

    public List<Double> getProbs() { return probs; }
    public void setProbs(List<Double> probs) { this.probs = probs; }
}

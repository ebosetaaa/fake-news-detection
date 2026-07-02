package com.example.fake_news_detection.service;

import com.example.fake_news_detection.dto.PredictRequest;
import com.example.fake_news_detection.dto.PredictResponse;
import com.example.fake_news_detection.model.PredictionRecord;
import com.example.fake_news_detection.repository.PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MLService {

    @Autowired
    private PredictionRepository predictionRepository;

    // Method your controller calls
    public PredictResponse getPrediction(PredictRequest request) {
        return predict(request.getText(), request.getShares(), request.getLikes(), request.getComments());
    }

    public PredictResponse predict(String text, double shares, double likes, double comments) {
        String prediction = "Fake";
        double fakeProb = 0.7;
        double realProb = 0.3;

        Map<String, Double> probs = new HashMap<>();
        probs.put("Fake", fakeProb);
        probs.put("Real", realProb);

        PredictionRecord record = new PredictionRecord(
                text,
                shares,
                likes,
                comments,
                prediction,
                probs
        );
        predictionRepository.save(record);

        return new PredictResponse(prediction, probs);
    }

    public PredictResponse predictFromList(String text, double shares, double likes, double comments, List<Double> probList) {
        Map<String, Double> probsMap = new HashMap<>();
        if (probList.size() >= 2) {
            probsMap.put("Fake", probList.get(0));
            probsMap.put("Real", probList.get(1));
        }

        String prediction = probsMap.get("Fake") >= probsMap.get("Real") ? "Fake" : "Real";

        PredictionRecord record = new PredictionRecord(
                text,
                shares,
                likes,
                comments,
                prediction,
                probsMap
        );
        predictionRepository.save(record);

        return new PredictResponse(prediction, probsMap);
    }
}
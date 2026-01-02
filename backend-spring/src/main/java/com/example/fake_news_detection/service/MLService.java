package com.example.fake_news_detection.service;

import com.example.fake_news_detection.dto.PredictRequest;
import com.example.fake_news_detection.dto.PredictResponse;
import com.example.fake_news_detection.model.PredictionRecord;
import com.example.fake_news_detection.repository.PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
public class MLService {

    private final RestTemplate restTemplate;
    private final PredictionRepository predictionRepository;
    private final String pythonApiUrl = "http://127.0.0.1:8000/predict";

    @Autowired
    public MLService(RestTemplate restTemplate, PredictionRepository predictionRepository) {
        this.restTemplate = restTemplate;
        this.predictionRepository = predictionRepository;
    }

    public PredictResponse getPrediction(PredictRequest request) {
        // Call Python FastAPI service
        Map<String, Object> response = restTemplate.postForObject(pythonApiUrl, request, Map.class);

        String pred = (String) response.get("prediction");
        @SuppressWarnings("unchecked")
        java.util.List<Double> probs = (java.util.List<Double>) response.get("probs");

        // Save to DB
        PredictionRecord record = new PredictionRecord(
                request.getText(),
                request.getShares(),
                request.getLikes(),
                request.getComments(),
                pred,
                probs
        );
        predictionRepository.save(record);

        return new PredictResponse(pred, probs);
    }
}

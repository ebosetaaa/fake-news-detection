package com.example.fake_news_detection.model;

import jakarta.persistence.*;
import java.util.Map;

@Entity
@Table(name = "predictions")
public class PredictionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String text;
    private double shares;
    private double likes;
    private double comments;
    private String prediction;

    @Convert(converter = JpaConverterJson.class)
    private Map<String, Double> probs;

    public PredictionRecord() {}

    public PredictionRecord(String text, double shares, double likes, double comments, String prediction, Map<String, Double> probs) {
        this.text = text;
        this.shares = shares;
        this.likes = likes;
        this.comments = comments;
        this.prediction = prediction;
        this.probs = probs;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }

    public double getShares() { return shares; }
    public void setShares(double shares) { this.shares = shares; }

    public double getLikes() { return likes; }
    public void setLikes(double likes) { this.likes = likes; }

    public double getComments() { return comments; }
    public void setComments(double comments) { this.comments = comments; }

    public String getPrediction() { return prediction; }
    public void setPrediction(String prediction) { this.prediction = prediction; }

    public Map<String, Double> getProbs() { return probs; }
    public void setProbs(Map<String, Double> probs) { this.probs = probs; }
}

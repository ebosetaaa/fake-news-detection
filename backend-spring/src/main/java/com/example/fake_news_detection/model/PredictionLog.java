package com.example.fake_news_detection.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
public class PredictionLog {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 4000)
    private String text;
    private double shares;
    private double likes;
    private double comments;
    private String prediction;

    @Column(length = 2000)
    private String probsJson;

    private LocalDateTime createdAt = LocalDateTime.now();

    // getters & setters
    // ... (generate in IDE)
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
    public String getProbsJson() { return probsJson; }
    public void setProbsJson(String probsJson) { this.probsJson = probsJson; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}

package com.example.fake_news_detection.dto;

public class PredictRequest {

    private String text;
    private double shares;
    private double likes;
    private double comments;

    public PredictRequest() {}

    public PredictRequest(String text, double shares, double likes, double comments) {
        this.text = text;
        this.shares = shares;
        this.likes = likes;
        this.comments = comments;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public double getShares() {
        return shares;
    }

    public void setShares(double shares) {
        this.shares = shares;
    }

    public double getLikes() {
        return likes;
    }

    public void setLikes(double likes) {
        this.likes = likes;
    }

    public double getComments() {
        return comments;
    }

    public void setComments(double comments) {
        this.comments = comments;
    }
}

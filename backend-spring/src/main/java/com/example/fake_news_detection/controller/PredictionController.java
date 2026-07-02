package com.example.fake_news_detection.controller;

import com.example.fake_news_detection.dto.PredictRequest;
import com.example.fake_news_detection.dto.PredictResponse;
import com.example.fake_news_detection.service.MLService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class PredictionController {

    @Autowired
    private MLService mlService;

    @GetMapping("/api/check")
    public String checkForm() {
        return "check"; // Thymeleaf template check.html
    }

    @PostMapping("/api/predict-form")
    public String predict(@RequestParam String text,
                          @RequestParam double shares,
                          @RequestParam double likes,
                          @RequestParam double comments,
                          Model model) {

        PredictRequest request = new PredictRequest(text, shares, likes, comments);
        PredictResponse response = mlService.getPrediction(request);

        model.addAttribute("result", response);
        return "result"; // Thymeleaf template result.html
    }
}

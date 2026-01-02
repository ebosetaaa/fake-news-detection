// src/main/java/com/example/fake_news_detection/repository/PredictionRepository.java
package com.example.fake_news_detection.repository;

import com.example.fake_news_detection.model.PredictionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PredictionRepository extends JpaRepository<PredictionRecord, Long> {
    // No need to declare save() — inherited from JpaRepository
}

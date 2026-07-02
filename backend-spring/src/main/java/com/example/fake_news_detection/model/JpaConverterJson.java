package com.example.fake_news_detection.model;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Converter
public class JpaConverterJson implements AttributeConverter<Map<String, Double>, String> {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public String convertToDatabaseColumn(Map<String, Double> attribute) {
        try {
            return objectMapper.writeValueAsString(attribute);
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }

    @Override
    public Map<String, Double> convertToEntityAttribute(String dbData) {
        try {
            return objectMapper.readValue(dbData, HashMap.class);
        } catch (IOException e) {
            return new HashMap<>();
        }
    }
}

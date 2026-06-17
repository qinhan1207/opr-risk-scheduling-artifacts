package com.qinhan.util;

import com.qinhan.model.PredictionResult;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * HTTP client for the Python OPR-TSMixer risk predictor.
 *
 * Java sends recent member-cluster-level latency observations. The Python side
 * constructs the 13-dimensional OPR feature sequence, applies the trained
 * OPR-TSMixer model, and returns the short-term placement-risk probability.
 */
public class OprRiskPredictor {

    private static final String MODEL_NAME = "OPR-TSMixer-v2";
    private static final String SERVER_URL = "http://127.0.0.1:5001/predict";
    private static final String WINDOW_SERVER_URL = "http://127.0.0.1:5001/predict/window";

    private OprRiskPredictor() {
    }

    public static PredictionResult predict(String clusterName, double currentLatency) {
        try {
            String jsonInput = String.format(
                    "{\"cluster_name\":\"%s\",\"latency\":%f}",
                    clusterName, currentLatency
            );
            return postJson(SERVER_URL, clusterName, jsonInput);
        } catch (Exception e) {
            return PredictionResult.failure(MODEL_NAME + " connection failed: " + e.getMessage());
        }
    }

    public static PredictionResult predictByLatencyWindow(String clusterName, List<Double> latencyWindow) {
        if (latencyWindow == null || latencyWindow.isEmpty()) {
            return PredictionResult.failure("Empty latencyWindow for " + MODEL_NAME);
        }

        try {
            StringBuilder windowJson = new StringBuilder("[");
            for (int i = 0; i < latencyWindow.size(); i++) {
                if (i > 0) {
                    windowJson.append(',');
                }
                windowJson.append(String.format("%f", latencyWindow.get(i)));
            }
            windowJson.append(']');

            String jsonInput = String.format(
                    "{\"cluster_name\":\"%s\",\"latency_window\":%s}",
                    clusterName,
                    windowJson
            );
            return postJson(WINDOW_SERVER_URL, clusterName, jsonInput);
        } catch (Exception e) {
            return PredictionResult.failure(MODEL_NAME + " connection failed: " + e.getMessage());
        }
    }

    private static PredictionResult postJson(String serverUrl, String clusterName, String jsonInput) throws Exception {
        URL url = new URL(serverUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection(Proxy.NO_PROXY);

        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setRequestProperty("X-Cluster-Name", clusterName);
        conn.setDoOutput(true);
        conn.setConnectTimeout(1000);
        conn.setReadTimeout(1000);

        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonInput.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        int code = conn.getResponseCode();
        if (code != 200 && code != 202) {
            return PredictionResult.failure("HTTP Error: " + code);
        }

        StringBuilder response = new StringBuilder();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                response.append(line.trim());
            }
        }

        return parseJson(response.toString());
    }

    private static PredictionResult parseJson(String json) {
        try {
            String cleanJson = json.replace(" ", "").replace("\n", "").replace("\"", "");
            boolean isFault = cleanJson.contains("is_fault:true");

            double prob = 0.0;
            int probIdx = cleanJson.indexOf("prob:");
            if (probIdx != -1) {
                int start = probIdx + 5;
                int endComma = cleanJson.indexOf(",", start);
                int endBrace = cleanJson.indexOf("}", start);

                int end = endComma;
                if (end == -1 || (endBrace != -1 && endBrace < endComma)) {
                    end = endBrace;
                }
                if (end != -1) {
                    try {
                        prob = Double.parseDouble(cleanJson.substring(start, end));
                    } catch (NumberFormatException ignored) {
                        prob = 0.0;
                    }
                }
            }

            String msg = "OK";
            if (cleanJson.contains("Buffering")) {
                msg = "Buffering...";
            }
            return new PredictionResult(true, isFault, prob, msg);
        } catch (Exception e) {
            return PredictionResult.failure("JSON Parse Error");
        }
    }
}

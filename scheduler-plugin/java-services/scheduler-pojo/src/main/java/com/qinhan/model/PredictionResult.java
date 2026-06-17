package com.qinhan.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Result returned by the Python OPR-TSMixer risk inference service.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResult {

    /** Whether the HTTP inference request succeeded. */
    private boolean isSuccess;

    /** Whether the predicted risk probability crosses the model threshold. */
    private boolean isFault;

    /** Short-term placement-risk probability in [0, 1]. */
    private double probability;

    /** Service-side message, for example "Buffering...". */
    private String message;

    public static PredictionResult failure(String errorMsg) {
        return new PredictionResult(false, false, 0.0, errorMsg);
    }

    @Override
    public String toString() {
        return String.format(
                "PredictionResult[success=%b, fault=%b, prob=%.4f, msg=%s]",
                isSuccess,
                isFault,
                probability,
                message
        );
    }
}

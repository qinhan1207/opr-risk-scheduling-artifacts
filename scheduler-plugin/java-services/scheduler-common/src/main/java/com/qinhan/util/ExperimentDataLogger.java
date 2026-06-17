package com.qinhan.util;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * CSV logger for scheduler scoring experiments.
 */
@Slf4j
public class ExperimentDataLogger {

    private static final String CSV_FILE = "experiment_results.csv";
    private static final SimpleDateFormat TIME_FMT = new SimpleDateFormat("HH:mm:ss");

    private ExperimentDataLogger() {
    }

    public static synchronized void log(
            String clusterName,
            long timestamp,
            double latency,
            double baselineScore,
            double rawLatencyScore,
            double networkOnlyScore,
            double oprTsmixerScore
    ) {
        File file = new File(CSV_FILE);
        boolean isNewFile = !file.exists();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
            if (isNewFile) {
                writer.write("TimeStr,Timestamp,Cluster,Real_Latency,Baseline,Raw_Latency,Network_Only,OPR_TSMixer");
                writer.newLine();
            }

            String line = String.format(
                    "%s,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f",
                    TIME_FMT.format(new Date(timestamp)),
                    timestamp,
                    clusterName,
                    latency,
                    baselineScore,
                    rawLatencyScore,
                    networkOnlyScore,
                    oprTsmixerScore
            );

            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            log.error("Failed to write scheduler experiment CSV: {}", e.getMessage());
        }
    }
}
